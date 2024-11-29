
#include <math.h>
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <stb_image_write.h>

#include <stdlib.h>
#include <onnxruntime_c_api.h>#

#undef DEBUG

typedef struct {
    int batch, channels, height, width;
} Shape;

typedef float* Array;

typedef struct {
    int num_batches;
    int embedding_dim;
} ArcShape;

typedef struct {
    Array output_tensor;
    ArcShape output_shape;
} ProcessResult;

Array resize_Image(unsigned char* image, int height, int width, int image_size, Shape* shape);

float cosine_simularity(ProcessResult* img_1,ProcessResult* img_2) {


    for(int i=0;i<512;i++) {
        printf("%f %f\n",img_1->output_tensor[i], img_2->output_tensor[i]);
    }
        float c=0,d=0,e=0;
        for(int i=0;i<512;i++) {
            e=e + img_2->output_tensor[i]*img_1->output_tensor[i];
            c=c + (float)pow(img_1->output_tensor[i],2);
            d=d + pow(img_2->output_tensor[i],2);
        }
    float x = c*d;
    float h=pow(x,0.5);
    float f=e/h;
    return f;
}

ProcessResult* process_image_rec(OrtSession* session, Array array, Shape shape) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtStatus* status = NULL;
    OrtMemoryInfo* memory_info = NULL;
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeCPU, &memory_info);

    OrtAllocator* allocator;
    status = g_ort->GetAllocatorWithDefaultOptions(&allocator);

    int64_t input_shape[] = {shape.batch, shape.channels, shape.height, shape.width};

    int total_elements = shape.height * shape.width * shape.channels;
    OrtValue* input_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(memory_info, array, total_elements * sizeof(float), input_shape,
                                                       4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    size_t num_input_nodes;
    status = g_ort->SessionGetInputCount(session, &num_input_nodes);
    char* input_name;
    status = g_ort->SessionGetInputName(session, 0, allocator, &input_name);

    size_t num_output_nodes;
    status = g_ort->SessionGetOutputCount(session, &num_output_nodes);
    char* output_name;
    status = g_ort->SessionGetOutputName(session, 0, allocator, &output_name);

    const char *input_names[] = {input_name};
    const char *output_names[] = {output_name};

    OrtValue* output_tensor = NULL;
    status = g_ort->Run(session, NULL, input_names, &input_tensor, 1, output_names, 1, &output_tensor);
    if(status!=NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        printf("Error: %s\n", msg);
    }
    OrtTensorTypeAndShapeInfo* output_info = NULL;
    status = g_ort->GetTensorTypeAndShape(output_tensor, &output_info);

    size_t dim_count;
    status = g_ort->GetDimensionsCount(output_info, &dim_count);

    int64_t* dimensions = malloc(sizeof(int64_t) * dim_count);
    status = g_ort->GetDimensions(output_info, dimensions, dim_count);
    if(status!=NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        printf("Error: %s\n", msg);
    }

    printf("Output tensor dimensions: ");
    for(size_t i = 0; i < dim_count; i++) {
        printf("%lld ", dimensions[i]);
    }

    ArcShape result_shape = {
        .num_batches = (int)dimensions[0],
        .embedding_dim = (int)dimensions[1],
    };

    float* output_data;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);


    size_t array_size = dimensions[0] * dimensions[1];
    Array result_array = malloc(sizeof(float) * array_size);
    memcpy(result_array, output_data, array_size * sizeof(float));



    ProcessResult* result = malloc(sizeof(ProcessResult));
    result->output_tensor = result_array;
    result->output_shape = result_shape;

    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);

    return result;
}

#ifdef DEBUG

void crop_save(unsigned char* image, Shape* shape, ProcessResult*  result) {
    unsigned char* cropped_image = malloc(shape->height * shape->width * 3);
    const int PERSON_CLASS = 0;

    for (int box = 0; box < result->output_shape.num_boxes; box++) {
        int offset = box * result->output_shape.values_per_box;


        float conf = 1.0f / (1.0f + exp(-result->output_tensor[offset + 4]));
        float class_conf = 1.0f / (1.0f + exp(-result->output_tensor[offset + 5 + PERSON_CLASS]));
        float final_conf = conf * class_conf;


        if (final_conf > 0.99) {

            float x = 1.0f / (1.0f + exp(-result->output_tensor[offset + 0]));
            float y = 1.0f / (1.0f + exp(-result->output_tensor[offset + 1]));
            float w = 1.0f / (1.0f + exp(-result->output_tensor[offset + 2]));
            float h = 1.0f / (1.0f + exp(-result->output_tensor[offset + 3]));


            int body_center_x = (int)(x * shape->width);
            int body_center_y = (int)(y * shape->height);
            int body_width = (int)(w * shape->width);
            int body_height = (int)(h * shape->height);


            float face_height = body_height * 0.2f;
            float face_width = face_height;

            int face_center_x = body_center_x;
            int face_center_y = (int)(body_center_y - body_height * 0.35f);

            float padding = 1.5f;
            int face_box_size = (int)(face_height * padding);


            int left = face_center_x - (face_box_size / 2);
            int right = face_center_x + (face_box_size / 2);
            int top = face_center_y - (face_box_size / 2);
            int bottom = face_center_y + (face_box_size / 2);


            printf("Person %d: conf=%.2f, body_center=(%d,%d), face_center=(%d,%d)\n",
                   box, final_conf, body_center_x, body_center_y, face_center_x, face_center_y);

            left = left < 0 ? 0 : left;
            top = top < 0 ? 0 : top;
            right = right >= shape->width ? shape->width - 1 : right;
            bottom = bottom >= shape->height ? shape->height - 1 : bottom;


            int crop_size = fmin(right - left, bottom - top);


            if (right - left > crop_size) {
                int diff = (right - left - crop_size) / 2;
                left += diff;
                right = left + crop_size;
            }
            if (bottom - top > crop_size) {
                int diff = (bottom - top - crop_size) / 2;
                top += diff;
                bottom = top + crop_size;
            }

            if (crop_size > 0) {
                stbir_resize_uint8_srgb(
                    image + (top * shape->width + left) * 3,
                    crop_size, crop_size, shape->width * 3,
                    cropped_image,
                    crop_size, crop_size, crop_size * 3,
                    3
                );

                char filename[100];
                sprintf(filename, "output/face_%.2f_%d.jpg", final_conf, box);
                stbi_write_jpg(filename, crop_size, crop_size, 3, cropped_image, 100);
            }
        }
    }

    free(cropped_image);
}

#endif

ProcessResult* final(char const* image_path, OrtSession* session) {
    int width, height, channels;
    unsigned char* image = stbi_load(image_path, &width, &height, &channels, 3);
    height = 200;
    width = 200;
    Shape* input_shape = malloc(sizeof(Shape));
    input_shape->batch = 1;
    input_shape->channels = channels;
    input_shape->height = height;
    input_shape->width = width;

    Shape resized_shape;
    Array input_array = resize_Image(image, height, width, 112, &resized_shape);

    ProcessResult* result = (ProcessResult*)process_image_rec(session, input_array, resized_shape);

    free(input_array);
    free(input_shape);

    return result;
}

int recognition() {

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);

    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);

    OrtSession* session;
    const wchar_t* model_path = L"w600k_r50.onnx";
    status = g_ort->CreateSession(env, model_path, session_options, &session);
    if(status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        printf("Error: %s\n", msg);
    }

    ProcessResult* img_1 = final("ktqqe6pm.png", session);
    ProcessResult* img_2 = final("ktqqe6pm.png", session);
    float prob = cosine_simularity(img_1,img_2);

    printf("Cosine Simularity: %f\n", prob);

    free(img_1);
    free(img_2);
}