# yoga_routes.py

from fastapi import APIRouter, Depends, HTTPException, Request # Import Request to access raw body
from sqlalchemy.orm import Session
from typing import List, Any # Use Any for type hints where schema was used

# Import your SQLAlchemy models
from models import models
# Import your database session dependency
from core.database import get_db
import asyncio # Though FastAPI handles the loop, good to have for async operations if needed
import numpy as np
import cv2 # OpenCV for image processing
import tensorflow as tf # TensorFlow Lite for model inference
import json # To send structured data (keypoints, feedback) over WS
import os # To check if file exists
from fastapi import WebSocket, WebSocketDisconnect # For WebSocket handling
# Removed import of Pydantic schemas from schema.yoga


interpreter = None
input_details = None
output_details = None

try:
    # Define the path to the TensorFlow Lite model file
    # Make sure this path is correct relative to where you run your main FastAPI app file
    model_path = os.path.join("routes", "movenet.tflite") # Adjust this path as needed
    if not os.path.exists(model_path):
         # Raise an error if the model file is not found
         raise FileNotFoundError(f"ML model file not found at: {model_path}")

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    # Allocate tensors to prepare the model for inference
    interpreter.allocate_tensors()
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Movenet model loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading Movenet model: {e}")
    print("Please ensure 'movenet.tflite' is in the correct directory relative to your application entry point.")
except Exception as e:
    print(f"❌ An unexpected error occurred while loading the Movenet model: {e}")
    # Set interpreter to None to indicate that the model is not available
    interpreter = None


# --- Pose Estimation Functions (Adapted from yoga_websocket.py) ---

def detect_pose_from_image_bytes(image_bytes: bytes) -> List[List[float]]:
    """
    Detects pose keypoints from image bytes using the loaded Movenet model.

    Args:
        image_bytes: The image data as bytes (e.g., from a webcam frame).

    Returns:
        A list of keypoints, where each keypoint is [y, x, confidence].
        Returns an empty list or raises an error if detection fails.

    Raises:
        RuntimeError: If the ML model was not loaded successfully.
        ValueError: If the image bytes are invalid and cannot be decoded.
        Exception: For other unexpected errors during detection.
    """
    if interpreter is None:
        # If the interpreter is None, the model failed to load at startup
        raise RuntimeError("ML model not loaded. Cannot perform pose detection.")

    try:
        # Convert bytes to a numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        # Decode the image using OpenCV
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data received. Could not decode.")

        # Convert the image from BGR (OpenCV default) to RGB (Movenet expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize the image to the Movenet input size (192x192)
        img_resized = cv2.resize(img_rgb, (192, 192))
        # Add a batch dimension to the image (Movenet expects input shape [1, height, width, channels])
        input_image = np.expand_dims(img_resized, axis=0).astype(np.uint8)

        # Set the input tensor for the interpreter
        interpreter.set_tensor(input_details[0]['index'], input_image)
        # Run the inference
        interpreter.invoke()

        # Get the output tensor (keypoints with scores)
        # The output shape is typically [1, 1, 17, 3] -> [batch_size, person_id, keypoint_id, [y, x, score]]
        # We take the keypoints for the first detected person
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])[0][0]
        # Convert the numpy array output to a standard Python list for JSON serialization
        return keypoints_with_scores.tolist()
    except Exception as e:
        # Catch any exceptions during the process and re-raise
        raise e

def detect_pose_from_filepath(filepath: str) -> List[List[float]]:
    """
    Detects pose keypoints from an image file path.

    Args:
        filepath: The local path to the image file.

    Returns:
        A list of keypoints, where each keypoint is [y, x, confidence].

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If the ML model is not loaded or fails during processing.
        Exception: For other unexpected errors.
    """
    try:
        # Check if the file exists before attempting to open
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Reference image file not found at: {filepath}")
        # Read the image file as bytes
        with open(filepath, "rb") as f:
            image_bytes = f.read()
        # Use the existing function to detect pose from bytes
        return detect_pose_from_image_bytes(image_bytes)
    except FileNotFoundError as e:
        # Re-raise FileNotFoundError directly
        raise e
    except RuntimeError as e:
         # Re-raise RuntimeError for ML model issues
         raise RuntimeError(f"ML model error processing reference image {filepath}: {e}")
    except Exception as e:
        # Catch any other exceptions and raise a general RuntimeError
        raise RuntimeError(f"Error processing reference image file {filepath}: {e}")


def compare_poses(pose1: List[List[float]], pose2: List[List[float]], confidence_threshold=0.3, similarity_threshold=0.05) -> str:
    """
    Compares two sets of pose keypoints and provides directional feedback.

    Args:
        pose1: The reference pose keypoints ([[y, x, confidence], ...]).
        pose2: The user's pose keypoints ([[y, x, confidence], ...]).
        confidence_threshold: Minimum confidence score for a keypoint to be considered.
        similarity_threshold: Minimum distance (in normalized coordinates) for feedback.

    Returns:
        A string containing feedback on pose differences, or "Perfect Pose!".
    """
    feedback = []
    # Movenet returns 17 keypoints. We iterate through them.
    # Keypoint format is [y, x, confidence] where x and y are normalized (0-1)
    body_parts = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]

    for i in range(17):
        # Check if both keypoints have sufficient confidence
        if pose1[i][2] > confidence_threshold and pose2[i][2] > confidence_threshold:
            # Get normalized coordinates
            y1, x1, _ = pose1[i]
            y2, x2, _ = pose2[i]

            # Calculate coordinate differences
            dx = x2 - x1 # positive dx means pose2's keypoint is more to the right (higher x)
            dy = y2 - y1 # positive dy means pose2's keypoint is more downwards (higher y)

            part_name = body_parts[i] if i < len(body_parts) else f"Keypoint {i}"
            part_feedback = []

            # Provide feedback based on significant differences in x and y coordinates
            if abs(dx) > similarity_threshold:
                 if dx > 0:
                     part_feedback.append("move your {} left".format(part_name)) # User needs to move part left to match ref
                 else: # dx < 0
                      part_feedback.append("move your {} right".format(part_name)) # User needs to move part right to match ref

            if abs(dy) > similarity_threshold:
                # Note: Y increases downwards in image coordinates.
                # So, positive dy means the user's keypoint is *lower* in the image than the reference.
                # Common yoga cues like "lift" or "lower" are usually relative to gravity.
                # We interpret positive dy (lower in image) as needing to "lift" the body part,
                # and negative dy (higher in image) as needing to "lower" the body part.
                # This interpretation might need adjustment based on the specific pose and camera angle.
                if dy > 0: # User keypoint is lower (higher y) than reference
                    part_feedback.append("lift your {}".format(part_name))
                else: # dy < 0 # User keypoint is higher (lower y) than reference
                     part_feedback.append("lower your {}".format(part_name))

            if part_feedback:
                # Join directional feedback for this body part
                feedback.append(f"{part_name}: {' and '.join(part_feedback)}")


    # Join feedback for all body parts into a single string
    return "Perfect Pose!" if not feedback else ", ".join(feedback)



# Create an API router for yoga-related endpoints
router = APIRouter()

# --- YogaData Endpoints ---

# Removed response_model and changed input type hint from Pydantic schema to dict/Any
@router.post("/data/")
async def create_yoga_data(request: Request, db: Session = Depends(get_db)):
    """
    Create a new Yoga Data entry (without schema validation).
    """
    # Get the raw JSON body as a dictionary
    yoga_data = await request.json()

    # Check for required fields manually
    if 'yoga_name' not in yoga_data or 'yoga_benefit' not in yoga_data:
         raise HTTPException(status_code=422, detail="Missing required fields: yoga_name, yoga_benefit")

    # Check if a yoga pose with the same name already exists
    db_yoga_data = db.query(models.YogaData).filter(models.YogaData.yoga_name == yoga_data.get('yoga_name')).first()
    if db_yoga_data:
        raise HTTPException(status_code=400, detail="Yoga pose with this name already exists")

    # Create a new YogaData model instance from the received data
    # Access data using .get() with a default value to handle missing optional fields gracefully
    db_yoga_data = models.YogaData(
        yoga_name=yoga_data.get('yoga_name'),
        yoga_benefit=yoga_data.get('yoga_benefit'),
        yoga_pose_url=yoga_data.get('yoga_pose_url'),
        yoga_time_in_sec=yoga_data.get('yoga_time_in_sec')
    )
    # Add the new entry to the database session
    db.add(db_yoga_data)
    # Commit the transaction
    db.commit()
    # Refresh the instance to get the generated ID
    db.refresh(db_yoga_data)
    # Return the created YogaData object (FastAPI will serialize it to JSON)
    return db_yoga_data

# Removed response_model
@router.get("/data/")
def read_yoga_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve a list of Yoga Data entries (without schema validation).
    """
    # Query the database for YogaData entries with pagination
    yoga_data = db.query(models.YogaData).offset(skip).limit(limit).all()
    # Return the list of YogaData objects (FastAPI will serialize them)
    return yoga_data

# Removed response_model
@router.get("/data/{yoga_data_id}")
def read_yoga_data_by_id(yoga_data_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a single Yoga Data entry by ID (without schema validation).
    """
    # Query the database for a specific YogaData entry by its ID
    yoga_data = db.query(models.YogaData).filter(models.YogaData.id == yoga_data_id).first()
    # If no entry is found, raise an HTTP 404 error
    if yoga_data is None:
        raise HTTPException(status_code=404, detail="Yoga Data not found")
    # Return the found YogaData object (FastAPI will serialize it)
    return yoga_data

# Removed response_model and changed input type hint
@router.put("/data/{yoga_data_id}")
async def update_yoga_data(yoga_data_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Update an existing Yoga Data entry by ID (without schema validation).
    """
    # Get the raw JSON body as a dictionary
    yoga_data_update = await request.json()

    # Query the database for the YogaData entry to update
    db_yoga_data = db.query(models.YogaData).filter(models.YogaData.id == yoga_data_id).first()
    # If no entry is found, raise an HTTP 404 error
    if db_yoga_data is None:
        raise HTTPException(status_code=404, detail="Yoga Data not found")

    # Check for duplicate yoga name if the name is being changed and provided in the update data
    if 'yoga_name' in yoga_data_update and db_yoga_data.yoga_name != yoga_data_update['yoga_name']:
         existing_yoga_data = db.query(models.YogaData).filter(models.YogaData.yoga_name == yoga_data_update['yoga_name']).first()
         if existing_yoga_data:
              raise HTTPException(status_code=400, detail="Yoga pose with this name already exists")

    # Update the fields of the database object with the new data
    # Only update fields that are present in the incoming request body
    if 'yoga_name' in yoga_data_update:
        db_yoga_data.yoga_name = yoga_data_update['yoga_name']
    if 'yoga_benefit' in yoga_data_update:
        db_yoga_data.yoga_benefit = yoga_data_update['yoga_benefit']
    if 'yoga_pose_url' in yoga_data_update:
        db_yoga_data.yoga_pose_url = yoga_data_update['yoga_pose_url']
    if 'yoga_time_in_sec' in yoga_data_update:
        db_yoga_data.yoga_time_in_sec = yoga_data_update['yoga_time_in_sec']

    # Commit the transaction
    db.commit()
    # Refresh the instance to get the updated data
    db.refresh(db_yoga_data)
    # Return the updated YogaData object (FastAPI will serialize it)
    return db_yoga_data

# Removed response_model
@router.delete("/data/{yoga_data_id}")
def delete_yoga_data(yoga_data_id: int, db: Session = Depends(get_db)):
    """
    Delete a Yoga Data entry by ID (without schema validation).
    """
    # Query the database for the YogaData entry to delete
    db_yoga_data = db.query(models.YogaData).filter(models.YogaData.id == yoga_data_id).first()
    # If no entry is found, raise an HTTP 404 error
    if db_yoga_data is None:
        raise HTTPException(status_code=404, detail="Yoga Data not found")

    # Delete the entry from the database session
    db.delete(db_yoga_data)
    # Commit the transaction
    db.commit()
    # Return a success message or the deleted object
    return {"detail": "Yoga Data deleted successfully", "deleted_item": db_yoga_data}


# --- YogaPoseLog Endpoints ---

# Removed response_model and changed input type hint
@router.post("/logs/")
async def create_yoga_pose_log(request: Request, db: Session = Depends(get_db)):
    """
    Create a new Yoga Pose Log entry (without schema validation).
    """
    # Get the raw JSON body as a dictionary
    yoga_log = await request.json()

    # Manually check for required fields and types
    required_fields = ['user_id', 'yoga_id', 'time', 'date']
    for field in required_fields:
        if field not in yoga_log:
             raise HTTPException(status_code=422, detail=f"Missing required field: {field}")

    # Basic type checking (you might need more robust validation)
    if not isinstance(yoga_log.get('user_id'), int) or not isinstance(yoga_log.get('yoga_id'), int):
         raise HTTPException(status_code=422, detail="user_id and yoga_id must be integers")
    if not isinstance(yoga_log.get('time'), str) or not isinstance(yoga_log.get('date'), str):
         raise HTTPException(status_code=422, detail="time and date must be strings")


    # Check if the user exists
    user = db.query(models.User).filter(models.User.id == yoga_log.get('user_id')).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if the yoga pose exists
    yoga_data = db.query(models.YogaData).filter(models.YogaData.id == yoga_log.get('yoga_id')).first()
    if yoga_data is None:
        raise HTTPException(status_code=404, detail="Yoga Data not found")

    # Create a new YogaPoseLog model instance
    db_yoga_log = models.YogaPoseLog(
        user_id=yoga_log.get('user_id'),
        yoga_id=yoga_log.get('yoga_id'),
        time=yoga_log.get('time'),
        date=yoga_log.get('date')
    )
    # Add the new log entry
    db.add(db_yoga_log)
    # Commit the transaction
    db.commit()
    # Refresh the instance to get the generated ID
    db.refresh(db_yoga_log)
    # Return the created log object (FastAPI will serialize it)
    return db_yoga_log

# Removed response_model
@router.get("/logs/")
def read_yoga_pose_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve a list of Yoga Pose Log entries (without schema validation).
    """
    # Query the database for YogaPoseLog entries with pagination
    yoga_logs = db.query(models.YogaPoseLog).offset(skip).limit(limit).all()
    # Return the list of log objects (FastAPI will serialize them)
    return yoga_logs

# Removed response_model
@router.get("/logs/{yoga_log_id}")
def read_yoga_pose_log_by_id(yoga_log_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a single Yoga Pose Log entry by ID (without schema validation).
    """
    # Query the database for a specific YogaPoseLog entry by its ID
    yoga_log = db.query(models.YogaPoseLog).filter(models.YogaPoseLog.id == yoga_log_id).first()
    # If no entry is found, raise an HTTP 404 error
    if yoga_log is None:
        raise HTTPException(status_code=404, detail="Yoga Pose Log not found")
    # Return the found log object (FastAPI will serialize it)
    return yoga_log

# Removed response_model
@router.get("/logs/user/{user_id}")
def read_yoga_pose_logs_by_user(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve Yoga Pose Log entries for a specific user (without schema validation).
    """
    # Check if the user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Query the database for YogaPoseLog entries filtered by user_id
    yoga_logs = db.query(models.YogaPoseLog).filter(models.YogaPoseLog.user_id == user_id).offset(skip).limit(limit).all()
    # Return the list of log objects for the user (FastAPI will serialize them)
    return yoga_logs

# Removed response_model and changed input type hint
@router.put("/logs/{yoga_log_id}")
async def update_yoga_pose_log(yoga_log_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Update an existing Yoga Pose Log entry by ID (without schema validation).
    """
    # Get the raw JSON body as a dictionary
    yoga_log_update = await request.json()

    # Query the database for the YogaPoseLog entry to update
    db_yoga_log = db.query(models.YogaPoseLog).filter(models.YogaPoseLog.id == yoga_log_id).first()
    # If no entry is found, raise an HTTP 404 error
    if db_yoga_log is None:
        raise HTTPException(status_code=404, detail="Yoga Pose Log not found")

    # Update the fields of the database object
    # Only update fields that are present in the incoming request body
    if 'user_id' in yoga_log_update:
        # Check if the user exists if user_id is being updated
        user = db.query(models.User).filter(models.User.id == yoga_log_update['user_id']).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        db_yoga_log.user_id = yoga_log_update['user_id']

    if 'yoga_id' in yoga_log_update:
        # Check if the yoga pose exists if yoga_id is being updated
        yoga_data = db.query(models.YogaData).filter(models.YogaData.id == yoga_log_update['yoga_id']).first()
        if yoga_data is None:
            raise HTTPException(status_code=404, detail="Yoga Data not found")
        db_yoga_log.yoga_id = yoga_log_update['yoga_id']

    if 'time' in yoga_log_update:
        db_yoga_log.time = yoga_log_update['time']

    if 'date' in yoga_log_update:
        db_yoga_log.date = yoga_log_update['date']

    # Commit the transaction
    db.commit()
    # Refresh the instance
    db.refresh(db_yoga_log)
    # Return the updated log object (FastAPI will serialize it)
    return db_yoga_log

# Removed response_model
@router.delete("/logs/{yoga_log_id}")
def delete_yoga_pose_log(yoga_log_id: int, db: Session = Depends(get_db)):
    """
    Delete a Yoga Pose Log entry by ID (without schema validation).
    """
    # Query the database for the YogaPoseLog entry to delete
    db_yoga_log = db.query(models.YogaPoseLog).filter(models.YogaPoseLog.id == yoga_log_id).first()
    # If no entry is found, raise an HTTP 404 error
    if db_yoga_log is None:
        raise HTTPException(status_code=404, detail="Yoga Pose Log not found")

    # Delete the entry
    db.delete(db_yoga_log)
    # Commit the transaction
    db.commit()
    # Return a success message or the deleted object
    return {"detail": "Yoga Pose Log deleted successfully", "deleted_item": db_yoga_log}


# --- Yoga Pose Comparison WebSocket Endpoint ---

@router.websocket("/ws/pose-compare/{yoga_data_id}")
async def websocket_pose_comparison(websocket: WebSocket, yoga_data_id: int, db: Session = Depends(get_db)):
    """
    WebSocket endpoint for real-time pose comparison.

    Clients connect to this endpoint specifying the ID of the reference yoga pose
    they want to compare against. Clients should send image bytes (e.g., from a webcam
    stream). The server will process the image, detect the user's pose, compare it
    to the reference pose, and send back feedback and keypoints as JSON.

    Args:
        websocket: The WebSocket connection object.
        yoga_data_id: The ID of the reference YogaData entry from the database.
        db: Database session dependency.
    """
    # Check if the ML model loaded successfully at startup
    if interpreter is None:
        # If the model didn't load, we cannot offer pose comparison.
        # Accept the connection briefly to send an error message before closing.
        await websocket.accept()
        await websocket.send_text("Error: ML model failed to load on the server. Pose comparison is unavailable.")
        await websocket.close()
        print(f"❌ WS connection failed for {yoga_data_id}: ML model not loaded.")
        return

    # 1. Establish WebSocket connection
    await websocket.accept()
    print(f"✅ WebSocket connection accepted for yoga_data_id: {yoga_data_id}")

    # 2. Load the reference pose from the database based on yoga_data_id
    reference_keypoints = None
    reference_pose_filepath = None
    try:
        # Retrieve the YogaData entry for the specified ID
        yoga_data = db.query(models.YogaData).filter(models.YogaData.id == yoga_data_id).first()
        if yoga_data is None:
            # If the YogaData entry is not found, close the connection
            await websocket.send_text(f"Error: Reference Yoga Data with id {yoga_data_id} not found.")
            await websocket.close()
            print(f"❌ WS connection closed for {yoga_data_id}: Yoga Data not found.")
            return

        # Get the local file path to the reference image from the database entry
        reference_pose_filepath = yoga_data.yoga_pose_url
        print(f"📸 Attempting to load reference pose from: {reference_pose_filepath}")
        # Load the reference image and detect its keypoints. This is done once per connection.
        reference_keypoints = detect_pose_from_filepath(reference_pose_filepath)
        print(f"📸 Reference pose loaded successfully for yoga_data_id {yoga_data_id}")

    except FileNotFoundError:
        # Handle cases where the reference image file does not exist
        error_msg = f"Error: Reference image file not found at {reference_pose_filepath}. Cannot start pose comparison."
        print(f"❌ WS connection closed for {yoga_data_id}: {error_msg}")
        await websocket.send_text(error_msg)
        await websocket.close()
        return
    except RuntimeError as e:
         # Handle errors during ML model processing of the reference image
         error_msg = f"Error loading/processing reference pose: {e}. Cannot start pose comparison."
         print(f"❌ WS connection closed for {yoga_data_id}: {error_msg}")
         await websocket.send_text(error_msg)
         await websocket.close()
         return
    except Exception as e:
        # Catch any other unexpected errors during the setup phase
        error_msg = f"An unexpected error occurred during setup for yoga_data_id {yoga_data_id}: {e}"
        print(f"❌ {error_msg}")
        await websocket.send_text(error_msg)
        await websocket.close()
        return


    # 3. Enter the message processing loop
    # This loop runs as long as the WebSocket connection is active
    try:
        while True:
            # Receive message from the client. We expect image bytes.
            try:
                message = await websocket.receive_bytes()
                # print(f"Received {len(message)} bytes from {yoga_data_id}.") # Debugging line
            except WebSocketDisconnect:
                # This exception is raised when the client closes the connection
                print(f"🔌 WebSocket disconnected for yoga_data_id: {yoga_data_id}")
                break # Exit the loop if the client disconnects
            except Exception as e:
                 # Handle errors during message reception
                 print(f"❌ Error receiving message for {yoga_data_id}: {e}")
                 # Send an error message back to the client but try to keep the connection open
                 await websocket.send_json({"error": f"Error receiving message: {e}"})
                 continue # Skip to the next iteration of the loop

            # Detect pose in the received image bytes
            user_keypoints = None
            try:
                user_keypoints = detect_pose_from_image_bytes(message)
                # print(f"Detected user keypoints for {yoga_data_id}.") # Debugging line
            except (ValueError, RuntimeError) as e:
                 # Handle specific errors from pose detection (invalid image, ML model issue)
                 print(f"❌ Error detecting user pose for {yoga_data_id}: {e}")
                 await websocket.send_json({"error": f"Error processing image: {e}"})
                 continue # Skip to the next iteration
            except Exception as e:
                 # Handle any other unexpected errors during pose detection
                 print(f"❌ Unexpected error during user pose detection for {yoga_data_id}: {e}")
                 await websocket.send_json({"error": f"An unexpected error occurred during pose detection: {e}"})
                 continue # Skip to the next iteration


            # Compare user's pose keypoints with the reference pose keypoints
            feedback = "Could not compare poses." # Default feedback in case of comparison error
            if user_keypoints is not None and reference_keypoints is not None:
                try:
                    feedback = compare_poses(reference_keypoints, user_keypoints)
                    # print(f"Generated feedback for {yoga_data_id}: {feedback[:100]}...") # Debugging line
                except Exception as e:
                    # Handle errors during the pose comparison process
                    print(f"❌ Error comparing poses for {yoga_data_id}: {e}")
                    feedback = f"Error comparing poses: {e}"


            # Prepare the response data to send back to the client
            response_data = {
                "feedback": feedback, # The comparison feedback string
                "user_keypoints": user_keypoints, # The detected keypoints for the user
                # Optionally, you could also send the reference_keypoints back if the client needs them,
                # but this might increase data transfer if many clients are connected.
                # "reference_keypoints": reference_keypoints
            }

            # Send the response data back to the client as JSON
            try:
                await websocket.send_json(response_data)
                # print(f"Sent response for {yoga_data_id}.") # Debugging line
            except Exception as e:
                # Handle errors during sending. This might indicate a broken connection.
                print(f"❌ Error sending response for {yoga_data_id}: {e}")
                # If sending fails, the connection is likely closing or already closed.
                # We can pass here and let the next receive_bytes call raise WebSocketDisconnect.
                pass # Continue loop, hoping the disconnect will be caught soon


    except WebSocketDisconnect:
        # This is the clean way to catch client-initiated disconnects after the loop exits
        print(f"🔌 Final WebSocket disconnect catch for yoga_data_id: {yoga_data_id}")
    except Exception as e:
        # Catch any unexpected errors that might occur within the main loop
        print(f"❌ Unexpected error in WebSocket loop for {yoga_data_id}: {e}")
        try:
            # Attempt to send a final error message before the connection fully closes
            await websocket.send_json({"error": f"An unexpected server error occurred: {e}"})
        except Exception:
            # If sending fails here, the connection is definitely closed
            pass # Connection might already be closed or sending failed

    finally:
        # This block executes when the WebSocket connection is closed for any reason (client disconnect, server error, etc.)
        print(f"🚪 WebSocket connection closed for yoga_data_id: {yoga_data_id}")
        # FastAPI/Starlette handles closing the underlying connection, so explicit close() might not be needed here
        # but could be added if specific cleanup was required.

