#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define EULER_NUMBER_F 2.71828182846

#include <stb_image.h>
#include <stb_image_resize2.h>
#include <stb_image_write.h>
#include <math.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

#undef DEBUG
#define test

typedef struct {
    int batch, channels, height, width;
} Shape;

typedef float* Array;

typedef struct {
    int num_boxes;
    int values_per_box;
} RetinaShape;

typedef struct {
    Array output_tensor;
    RetinaShape output_shape;
} ProcessResult;

Array resize_Image(unsigned char* image, int height, int width, int image_size, Shape* shape) {

    if(!image) {
        return NULL;
    }
    unsigned char* resized_image = malloc(image_size * image_size * 3);
    stbir_resize_uint8_srgb(image, width, height, 0, resized_image, image_size, image_size, 0, 3);

    shape->batch = 1;
    shape->channels = 3;
    shape->height = image_size;
    shape->width = image_size;

    int total_elements =  shape->height * shape->width * shape->channels;

    Array array = malloc(total_elements * sizeof(float));
    for(int i = 0; i < total_elements; ++i) {
        float x = (float)resized_image[i];
        array[i] = x / 255.0f;
    }
    //stbi_write_jpg("lol.jpg", 640, 640, 3, array, 100);

    free(resized_image);
    return array;
}

float* pad_and_resize_image(unsigned char* image, int height, int width, int target_size, Shape* shape) {
    if (!image) {
        return NULL;
    }

    // Calculate padding to make the image square
    int max_dim = height > width ? height : width;
    int pad_height = max_dim - height;
    int pad_width = max_dim - width;
    int pad_top = pad_height / 2;
    int pad_left = pad_width / 2;

    unsigned char* padded_image = malloc(max_dim * max_dim * 3);
    if (!padded_image) {
        fprintf(stderr, "Memory allocation failed for padded_image\n");
        return NULL;
    }
    memset(padded_image, 128, max_dim * max_dim * 3); // Fill with gray (128)

    // Copy original image to center of padded image
    for (int y = 0; y < height; y++) {
        memcpy(
            padded_image + ((y + pad_top) * max_dim + pad_left) * 3,
            image + y * width * 3,
            width * 3
        );
    }

    // Allocate memory for resized image
    int total_elements = target_size * target_size * 3;
    unsigned char* resized_image = malloc(total_elements);
    if (!resized_image) {
        fprintf(stderr, "Memory allocation failed for resized_image\n");
        free(padded_image);
        return NULL;
    }

    // Resize the image
    stbir_resize_uint8_srgb(
        padded_image, max_dim, max_dim, 0,
        resized_image, target_size, target_size, 0,
        3
    );


    float* resized_image_ptr = malloc(total_elements * sizeof(float));
    if (!resized_image_ptr) {
        fprintf(stderr, "Memory allocation failed for resized_image_ptr\n");
        free(padded_image);
        free(resized_image);
        return NULL;
    }

    for (int i = 0; i < total_elements; i++) {
        resized_image_ptr[i] = (float)resized_image[i] / 255.0f;
    }

    // Update shape metadata
    shape->batch = 1;
    shape->channels = 3;
    shape->height = target_size;
    shape->width = target_size;

    // Debugging
    #ifdef DEBUG
    unsigned char* debug_image = malloc(total_elements);
    for (int i = 0; i < total_elements; i++) {
        debug_image[i] = (unsigned char)(resized_image_ptr[i] * 255.0f);
    }
    stbi_write_jpg("debug_padded_resized.jpg", target_size, target_size, 3, debug_image, 100);
    free(debug_image);
    #endif

    free(padded_image);
    free(resized_image);

    return resized_image_ptr;
}


ProcessResult* process_image(OrtSession* session, Array array, Shape shape) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtStatus* status = NULL;
    OrtMemoryInfo* memory_info = NULL;

    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeCPU, &memory_info);
    if (status != NULL) {
        printf("Error creating memory info: %s\n", g_ort->GetErrorMessage(status));
        return NULL;
    }

    int64_t input_shape[] = {shape.batch, shape.channels, shape.height, shape.width};
    int total_elements = shape.height * shape.width * shape.channels;

    OrtValue* input_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, array, total_elements * sizeof(float), input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor
    );
    if (status != NULL) {
        printf("Error creating input tensor: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseMemoryInfo(memory_info);
        return NULL;
    }

    OrtAllocator* allocator = NULL;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    char* input_name = NULL;
    char* output_name = NULL;

    status = g_ort->SessionGetInputName(session, 0, allocator, &input_name);
    if (status != NULL) {
        printf("Error getting input name: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        return NULL;
    }

    size_t input_num;
    status = g_ort->SessionGetInputCount(session, &input_num);

    size_t output_num;
    status = g_ort->SessionGetOutputCount(session, &output_num);

    const char* input_names[] = {input_name};
    const char* output_names[9];
    for(int i = 0; i < output_num; i++){
        status = g_ort->SessionGetOutputName(session, i, allocator, &output_name);
        if (status != NULL) {
            printf("Error getting output name: %s\n", g_ort->GetErrorMessage(status));
            g_ort->ReleaseValue(input_tensor);
            g_ort->ReleaseMemoryInfo(memory_info);
            g_ort->AllocatorFree(allocator, input_name);
            return NULL;
        }

        output_names[i] = output_name;
        printf("Output name: %s\n", output_names[i]);
    }

    OrtValue** output_tensors = malloc(output_num * sizeof(OrtValue*));
    for(size_t i = 0; i < output_num; i++) {
        output_tensors[i] = NULL;
    }

    status = g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1,
                        (const char* const*)output_names, output_num, output_tensors);
    if (status != NULL) {
        printf("Error running session: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->AllocatorFree(allocator, input_name);
        g_ort->AllocatorFree(allocator, output_name);
        return NULL;
    }
    OrtTensorTypeAndShapeInfo* output_info = NULL;
    g_ort->GetTensorTypeAndShape(output_tensors[7], &output_info);
    size_t dim_count;
    g_ort->GetDimensionsCount(output_info, &dim_count);

    int64_t* dimensions = malloc(sizeof(int64_t) * dim_count);
    g_ort->GetDimensions(output_info, dimensions, dim_count);

    printf("Number of dimensions: %llu\n", dim_count);
    printf("Output tensor dimensions: ");
    for(size_t i = 0; i < dim_count; i++) {
        printf("%lld ", dimensions[i]);
    }

    float* output_data = NULL;
    g_ort->GetTensorMutableData(output_tensors[7], (void**)&output_data);

    size_t array_size = dimensions[0] * dimensions[1];
    Array result_array = malloc(array_size * sizeof(float));
    memcpy(result_array, output_data, array_size * sizeof(float));

    RetinaShape result_shape = {
        .num_boxes = (int)dimensions[1],
        .values_per_box = (int)dimensions[0]
    };

    ProcessResult* result = malloc(sizeof(ProcessResult));
    result->output_tensor = result_array;
    result->output_shape = result_shape;

    free(dimensions);
    g_ort->ReleaseValue(input_tensor);
    for(size_t i = 0; i < output_num; i++) {
        if(output_tensors[i] != NULL) {
            g_ort->ReleaseValue(output_tensors[i]);
        }
    }
    free(output_tensors);
    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->AllocatorFree(allocator, input_name);
    g_ort->AllocatorFree(allocator, output_name);

    return result;
}



void crop_save(unsigned char* image, Shape* shape, ProcessResult* result) {
    unsigned char* cropped_image = malloc(shape->height * shape->width * 3);

    printf("Image dimensions: %d x %d\n", shape->width, shape->height);
    float sum = 0.0f;
    float max = 0.0f;
    for (int i = 0; i < result->output_shape.num_boxes; i++) {
        int offset = i * result->output_shape.values_per_box;
        float conf = result->output_tensor[offset + 4];
        sum += exp(conf);
        if (conf > max) {
            max = conf;
        }
    }

    printf("%f\n", max);
    for (int box = 0; box < result->output_shape.values_per_box; box++) {
        int offset = box * result->output_shape.num_boxes;
        float conf = 1.0f / (1.0f + exp(-result->output_tensor[offset + 4]));
        //float conf = exp(result->output_tensor[offset + 4]) / sum;
        if(conf > 0.9) {


            float x = result->output_tensor[offset + 0];
            float y = result->output_tensor[offset + 1];
            float w = result->output_tensor[offset + 2];
            float h = result->output_tensor[offset + 3];


            printf("\nBox %d normalized values:\n", box);
            printf("x: %f, y: %f, w: %f, h: %f, conf: %f\n", x, y, w, h, conf);

            int center_x = (int)(x * shape->width / 640.f);
            int center_y = (int)(y * shape->height / 640.f);
            int face_width = (int)(w * shape->width / 640.f);
            int face_height = (int)(h * shape->height / 640.f);

            int left = center_x - face_width/2;
            int top = center_y - face_height/2;
            int right = left + face_width;
            int bottom = top + face_height;

            // Ensure boundaries are within image dimensions
            left = max(0, left);
            top = max(0, top);
            right = min(shape->width, right);
            bottom = min(shape->height, bottom);

            // Calculate final width and height
            int final_width = right - left;
            int final_height = bottom - top;

            // Ensure square crop fits within bounds
            int crop_size = max(final_width, final_height);

            // Adjust for square crop
            if (final_width < crop_size) {
                int diff = crop_size - final_width;
                left = max(0, left - diff / 2);
                right = min(shape->width, left + crop_size);
            }

            if (final_height < crop_size) {
                int diff = crop_size - final_height;
                top = max(0, top - diff / 2);
                bottom = min(shape->height, top + crop_size);
            }

            // Final clamping after square adjustment
            right = min(shape->width, left + crop_size);
            bottom = min(shape->height, top + crop_size);

            if (final_width > 0 && final_height > 0) {
                // Create square crop by taking larger dimension
                int crop_size = max(final_width, final_height);

                // Adjust boundaries to center the face in the square
                if (final_width < crop_size) {
                    int diff = crop_size - final_width;
                    left = max(0, left - diff/2);
                    right = min(shape->width, left + crop_size);
                }
                if (final_height < crop_size) {
                    int diff = crop_size - final_height;
                    top = max(0, top - diff/2);
                    bottom = min(shape->height, top + crop_size);
                }

                // Final boundary check
                crop_size = min(right - left, bottom - top);

                printf("Crop coordinates for face %d:\n", box);
                printf("left=%d, top=%d, right=%d, bottom=%d, size=%d\n",
                       left, top, right, bottom, crop_size);

                int cropped_width = right - left;
                int cropped_height = bottom - top;

                unsigned char* temp_crop = malloc(cropped_width * cropped_height * 3);

                for (int row = 0; row < cropped_height; row++) {
                    memcpy(
                        temp_crop + row * cropped_width * 3,
                        image + ((top + row) * shape->width + left) * 3,
                        cropped_width * 3
                    );
                }


                // Copy and resize the cropped region
                stbir_resize_uint8_srgb(
                    temp_crop, cropped_width, cropped_height, cropped_width * 3,
                    cropped_image, crop_size, crop_size, crop_size * 3,
                    3
                );

                char filename[100];
                sprintf(filename, "output/face_%d.jpg", box);
                stbi_write_jpg(filename, crop_size, crop_size, 3, cropped_image, 100);
                printf("Saved face_%d.jpg\n", box);
            }
        }
    }

    free(cropped_image);
}


int main() {
    int width, height, channels;
    unsigned char* image = stbi_load("crowd.jpg", &width, &height, &channels, 3);
    Shape* input_shape = malloc(sizeof(Shape));
    input_shape->batch = 1;
    input_shape->channels = channels;
    input_shape->height = height;
    input_shape->width = width;

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);

    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);

    OrtSession* session;
    const wchar_t* model_path = L"det_10g.onnx";
    status = g_ort->CreateSession(env, model_path, session_options, &session);
    if(status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        printf("Error: %s\n", msg);
    }

    Shape resized_shape;
    Array resized_array = pad_and_resize_image(image, height, width, 640, &resized_shape);

    ProcessResult* result = (ProcessResult*)process_image(session, resized_array, resized_shape);


    //crop_save(image, input_shape, result);

    free(resized_array);
    free(result->output_tensor);
    free(result);

}
