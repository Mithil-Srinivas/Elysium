#include <gtk/gtk.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define IMAGE_INTERVAL 500  // Time in milliseconds between images
#define TOTAL_IMAGES 69     // Total images to capture

GtkEntry *entry_fields[2];
GtkWidget *attendance_file_label;
GtkCssProvider *css_provider;

// Function to adjust font sizes dynamically
void adjust_font_size(GtkWidget *widget, GdkRectangle *allocation, gpointer user_data) {
    int width = allocation->width;
    int height = allocation->height;

    // Scale font size dynamically based on window dimensions
    int font_size = (width + height) / 100;
    int header_font_size = font_size * 3;  // Increased size for ELYSIUM header

    char css[2048];
    snprintf(css, sizeof(css),
             "window { background-color: black; } "
             "#left_box, #right_box { background-color: black; border: 3px solid white; padding: 20px; border-radius: 15px; } "
             "label { color: black; font-size: %dpx; } "
             "#header_label { color: white; font-size: %dpx; font-weight: bold; } "
             "#attendance_label, #new_registration_label { color: white; font-size: %dpx; font-weight: bold; } "
             "#id_label, #name_label { color: white; font-size: %dpx; } "
             "button { background-color: white; color: black; font-size: %dpx; padding: 10px 20px; } "
             "entry { font-size: %dpx; padding: 10px;}",
             font_size, header_font_size, font_size * 2, font_size, font_size, font_size);

    gtk_css_provider_load_from_data(css_provider, css, -1, NULL);
}

// Callback for file upload button
void on_upload_button_clicked(GtkButton *button, gpointer user_data) {
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Select Attendance File", NULL, GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open", GTK_RESPONSE_ACCEPT,
        NULL);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *file_path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        const char *destination_path = "C:\\Users\\Administrator\\Desktop\\Codes\\AttendanceSystem\\Attendance";

        char command[512];
        snprintf(command, sizeof(command), "copy \"%s\" \"%s\"", file_path, destination_path);
        system(command);

        gtk_label_set_text(GTK_LABEL(attendance_file_label), "Attendance uploaded successfully!");
        g_free(file_path);
    }

    gtk_widget_destroy(dialog);
}

// Callback for register button
void on_register_button_clicked(GtkButton *button, gpointer user_data) {
    const char *id = gtk_entry_get_text(GTK_ENTRY(entry_fields[0]));
    const char *name = gtk_entry_get_text(GTK_ENTRY(entry_fields[1]));

    if (strlen(id) == 0 || strlen(name) == 0) {
        GtkWidget *dialog = gtk_message_dialog_new(NULL, GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_ERROR, GTK_BUTTONS_OK, "Both ID and Name must be filled out.");
        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
        return;
    }

    char folder_name[256];
    snprintf(folder_name, sizeof(folder_name), "ImageTraining/%s - %s", id, name);

    mkdir("ImageTraining");
    mkdir(folder_name);

    char command[512];
    snprintf(command, sizeof(command),
             "ffmpeg -f dshow -i video=\"Integrated Webcam\" "
             "-vf \"drawtext=text='Capturing...':fontcolor=white:x=10:y=10\" "
             "-r 5 -frames:v %d -q:v 2 -y \"%s/image%%02d.jpg\"",
             TOTAL_IMAGES, folder_name);

    system(command);
}

// GUI Initialization
void initialize_gui(int argc, char *argv[]) {
    GtkWidget *window, *main_grid, *header_label, *left_box, *right_box;
    GtkWidget *attendance_label, *upload_button;
    GtkWidget *id_label, *name_label, *id_entry, *name_entry, *register_button;
    GtkWidget *id_row, *name_row, *new_registration_label;

    gtk_init(&argc, &argv);

    // Create main window
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "ELYSIUM");
    gtk_window_maximize(GTK_WINDOW(window));
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    // Create CSS provider for dynamic styling
    css_provider = gtk_css_provider_new();
    GtkStyleContext *context = gtk_widget_get_style_context(window);
    gtk_style_context_add_provider_for_screen(
        gdk_screen_get_default(),
        GTK_STYLE_PROVIDER(css_provider),
        GTK_STYLE_PROVIDER_PRIORITY_USER);

    // Create main grid
    main_grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(main_grid), 20);
    gtk_grid_set_column_spacing(GTK_GRID(main_grid), 50);
    gtk_widget_set_halign(main_grid, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(main_grid, GTK_ALIGN_CENTER);
    gtk_container_add(GTK_CONTAINER(window), main_grid);

    // Add header
    header_label = gtk_label_new("ELYSIUM");
    gtk_widget_set_name(header_label, "header_label");
    gtk_grid_attach(GTK_GRID(main_grid), header_label, 0, 0, 2, 1);

    // Left box
    left_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 20);
    gtk_widget_set_name(left_box, "left_box");
    gtk_widget_set_size_request(left_box, 300, -1);  // Increase the width of the left box

    attendance_label = gtk_label_new("Upload Video File");
    gtk_widget_set_name(attendance_label, "attendance_label");

    upload_button = gtk_button_new_with_label("Upload Video File");
    attendance_file_label = gtk_label_new("");

    // Adjust the size and padding of the button
    gtk_widget_set_size_request(upload_button, 120, 40);  // Set a smaller size
    gtk_box_pack_start(GTK_BOX(left_box), attendance_label, FALSE, FALSE, 10);
    gtk_box_pack_start(GTK_BOX(left_box), upload_button, FALSE, FALSE, 20); // Add more space below
    gtk_box_pack_start(GTK_BOX(left_box), attendance_file_label, FALSE, FALSE, 10);

    g_signal_connect(upload_button, "clicked", G_CALLBACK(on_upload_button_clicked), NULL);

    // Right box
    right_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 20);
    gtk_widget_set_name(right_box, "right_box");

    new_registration_label = gtk_label_new("New Registration");
    gtk_widget_set_name(new_registration_label, "new_registration_label");

    id_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    id_label = gtk_label_new("ID:");
    gtk_widget_set_name(id_label, "id_label");
    id_entry = gtk_entry_new();
    gtk_box_pack_start(GTK_BOX(id_row), id_label, FALSE, FALSE, 10);
    gtk_box_pack_start(GTK_BOX(id_row), id_entry, TRUE, TRUE, 10);

    name_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    name_label = gtk_label_new("Name:");
    gtk_widget_set_name(name_label, "name_label");
    name_entry = gtk_entry_new();
    gtk_box_pack_start(GTK_BOX(name_row), name_label, FALSE, FALSE, 10);
    gtk_box_pack_start(GTK_BOX(name_row), name_entry, TRUE, TRUE, 10);

    register_button = gtk_button_new_with_label("Register");
    g_signal_connect(register_button, "clicked", G_CALLBACK(on_register_button_clicked), NULL);

    gtk_box_pack_start(GTK_BOX(right_box), new_registration_label, FALSE, FALSE, 10);
    gtk_box_pack_start(GTK_BOX(right_box), id_row, FALSE, FALSE, 10);
    gtk_box_pack_start(GTK_BOX(right_box), name_row, FALSE, FALSE, 10);
    gtk_box_pack_start(GTK_BOX(right_box), register_button, FALSE, FALSE, 10);

    // Add boxes to main grid
    gtk_grid_attach(GTK_GRID(main_grid), left_box, 0, 1, 1, 1);
    gtk_grid_attach(GTK_GRID(main_grid), right_box, 1, 1, 1, 1);

    // Connect window resize to adjust font size
    g_signal_connect(window, "size-allocate", G_CALLBACK(adjust_font_size), NULL);

    gtk_widget_show_all(window);
    gtk_main();
}

// Main function
int main(int argc, char *argv[]) {
    initialize_gui(argc, argv);
    return 0;
}
