import customtkinter as ctk
import cv2
import time
from PIL import Image
from ultralytics import YOLO

# Initialize the YOLOv8 model with custom classes
model = YOLO('best.pt')  # Replace with your custom model path
target_classes = {"Back_Palm", "Front_Palm", "Hair"}  # Custom classes to track

# Initialize variables for detection timing
start_time = None
elapsed_time = 0

# Initialize the customtkinter application
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# Function to display the detection page
def open_detection_page():
    global cap, start_time, elapsed_time
    cap = cv2.VideoCapture(0)  # Start webcam capture
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    # Temporarily disable the main window
    root.withdraw()

    # Create a new window for detection
    detection_window = ctk.CTkToplevel()
    detection_window.geometry("900x600")
    detection_window.title("Live Detection")

    # Label to display video feed
    video_label = ctk.CTkLabel(
        detection_window,
        text="", 
        width=640, 
        height=360
    )
    video_label.place(x=10, y=10)  # Top-left live feed

    # Label to display the detection timer
    timer_label = ctk.CTkLabel(
        detection_window,
        text="Detection Timer: 0s",
        font=("Arial", 18),
        width=200,
        height=30
    )
    timer_label.place(x=10, y=400)  # Timer display below live feed

    # Function to update the video frame and timer
    def update_frame():
        global cap, start_time, elapsed_time

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Perform object detection
                results = model(frame)

                # Extract detected class names
                detected_classes = {
                    model.names[int(result[5])] for result in results[0].boxes.data.cpu().numpy()
                }

                # Check if any of the target classes are detected
                if target_classes.intersection(detected_classes):
                    if start_time is None:
                        start_time = time.time()  # Start timer
                    else:
                        elapsed_time += time.time() - start_time
                        start_time = time.time()  # Reset timer
                else:
                    start_time = None  # Pause timer

                # Render the results on the frame
                frame_with_boxes = results[0].plot()

                # Resize the frame for display in the GUI
                frame_with_boxes = cv2.resize(frame_with_boxes, (640, 360))

                # Convert image from OpenCV BGR to RGB format
                image = Image.fromarray(cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB))

                # Convert PIL image to CTkImage
                photo = ctk.CTkImage(image, size=(640, 360))

                # Update the video label with the image
                video_label.imgtk = photo  # Keep a reference to avoid garbage collection
                video_label.configure(image=photo)

                # Update the timer label
                timer_label.configure(
                    text=f"Detection Timer: {int(elapsed_time)}s"
                )

            # Schedule the next frame update
            detection_window.after(10, update_frame)

    update_frame()

    # Function to stop detection and return to the home page
    def stop_detection():
        global cap, start_time, elapsed_time
        if cap.isOpened():
            cap.release()
        detection_window.destroy()
        root.deiconify()  # Re-enable the main window

    # Bind the 'q' key to stop detection
    detection_window.bind("<Key-q>", lambda event: stop_detection())


# Create the main application window (Home Page)
root = ctk.CTk()
root.geometry("900x600")
root.title("Safety Detection of Workers")

# Add a heading label
heading_label = ctk.CTkLabel(
    root, 
    text="Safety Detection of Workers", 
    font=("Arial", 24, "bold")
)
heading_label.pack(pady=40)

# Add a button to start detection
start_button = ctk.CTkButton(
    root, 
    text="Start Detection", 
    font=("Arial", 16), 
    command=open_detection_page
)
start_button.pack(pady=20)

# Start the main loop
root.mainloop()

# Release the video capture when the application is closed
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()






