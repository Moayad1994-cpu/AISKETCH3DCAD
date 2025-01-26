import os
import cv2
import numpy as np
import ezdxf
import logging
import sys
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template

app = Flask(__name__)

# Global processing status
processing_status = {
    'processing': False,
    'progress': 0,
    'message': 'Idle',
    'error': None
}

def setup_logging(base_path):
    """Set up logging configuration."""
    log_dir = os.path.join(base_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'processing_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


class VideoToDXFConverter:
    def __init__(self, base_path, dxf_folder):
        """
        :param base_path: The path where uploads are stored
        :param dxf_folder: Unique folder name inside dxf_output for this video
        """
        self.base_path = base_path
        self.dxf_output_dir = os.path.join(self.base_path, 'dxf_output', dxf_folder)
        os.makedirs(self.dxf_output_dir, exist_ok=True)

        self.processed_dir = os.path.join(self.base_path, "processed_frames")
        os.makedirs(self.processed_dir, exist_ok=True)

        self.logger = setup_logging(self.base_path)

        # Initialize parameters
        self.edge_params = {
            'threshold1': 50,
            'threshold2': 150,
            'apertureSize': 3
        }
        self.frame_interval = 30

    def extract_video_frames(self, video_path):
        """Extract frames from video file."""
        try:
            self.logger.info(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved_frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_interval == 0:
                    frame_path = os.path.join(
                        self.processed_dir,
                        f"frame_{frame_count}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    saved_frames.append(frame_path)
                    self.logger.info(f"Saved frame: {frame_path}")

                frame_count += 1

            cap.release()
            return saved_frames

        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            return []

    def detect_edges(self, image_path):
        """Detect edges in image."""
        try:
            self.logger.info(f"Processing image: {image_path}")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(
                blurred,
                self.edge_params['threshold1'],
                self.edge_params['threshold2'],
                apertureSize=self.edge_params['apertureSize']
            )
            return edges

        except Exception as e:
            self.logger.error(f"Edge detection failed: {str(e)}")
            return None

    def create_dxf(self, edges, dxf_path, offset=(0, 0)):
        """Create DXF file from edges."""
        try:
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()

            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                points = []
                for point in contour:
                    x = float(point[0][0]) + offset[0]
                    # Flip y so the bottom-left is origin
                    y = float(edges.shape[0] - point[0][1]) + offset[1]
                    points.append((x, y))

                if len(points) > 2:
                    msp.add_lwpolyline(points, close=True)

            doc.saveas(dxf_path)
            self.logger.info(f"DXF file created: {dxf_path}")
            return True

        except Exception as e:
            self.logger.error(f"DXF creation failed: {str(e)}")
            return False

    def convert_video_to_dxf(self, video_file):
        """Convert the given video file to one or more DXF files."""
        processing_status['processing'] = True
        processing_status['progress'] = 0
        processing_status['error'] = None
        processing_status['message'] = 'Extracting frames...'
        start_time = time.time()

        frames = self.extract_video_frames(video_file)
        total_frames = len(frames)
        if total_frames == 0:
            # No frames extracted
            processing_status['processing'] = False
            processing_status['message'] = 'No frames extracted'
            return

        for i, frame_path in enumerate(frames):
            # Update progress between 0 and 100
            # We'll mark up to 90% for normal processing, last 10% for final tasks
            # i / total_frames -> fraction of frames done
            fraction = (i + 1) / total_frames
            processing_status['progress'] = int(fraction * 90)

            # Calculate an ETA
            elapsed = time.time() - start_time
            avg_per_frame = elapsed / (i + 1)
            frames_left = total_frames - (i + 1)
            eta_seconds = int(avg_per_frame * frames_left)

            processing_status['message'] = (
                f"Processing frame {i+1} of {total_frames}... "
                f"ETA: {eta_seconds}s"
            )

            edges = self.detect_edges(frame_path)
            if edges is not None:
                base_name = os.path.basename(video_file)
                dxf_file_name = f"{os.path.splitext(base_name)[0]}_frame_{i}.dxf"
                dxf_path = os.path.join(self.dxf_output_dir, dxf_file_name)
                self.create_dxf(edges, dxf_path, offset=(i * 1000, 0))

        # Finalize progress
        processing_status['progress'] = 100
        processing_status['message'] = 'Processing complete!'
        processing_status['processing'] = False


@app.route('/')
def index():
    # Render index.html from a templates folder if you placed it in "templates/index.html".
    # Or directly serve it if you placed the HTML in the same directory:
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        uploaded_file = request.files.get('file')
        if not uploaded_file:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        filename = uploaded_file.filename
        if filename == '':
            return jsonify({'success': False, 'error': 'Invalid file name'}), 400

        # Create an uploads folder if not exist
        upload_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(upload_dir, filename)
        uploaded_file.save(file_path)

        # Create unique subfolder in dxf_output for this video
        # (timestamp + original filename's base)
        unique_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_no_ext = os.path.splitext(filename)[0]
        dxf_folder_name = f"{unique_id}_{base_no_ext}"

        # Start the converter
        converter = VideoToDXFConverter(upload_dir, dxf_folder_name)
        converter.convert_video_to_dxf(file_path)

        # Return folder name so client can request only new files
        return jsonify({'success': True, 'dxf_folder': dxf_folder_name}), 200

    except Exception as e:
        processing_status['processing'] = False
        processing_status['error'] = str(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    return jsonify(processing_status)

@app.route('/list-dxf-files', methods=['GET'])
def list_dxf_files():
    folder = request.args.get('folder', '')
    base_folder = os.path.join(os.getcwd(), 'uploads', 'dxf_output')

    # If a specific folder is requested, list only that folder
    dxf_folder = os.path.join(base_folder, folder) if folder else base_folder

    files = []
    if os.path.isdir(dxf_folder):
        for f in os.listdir(dxf_folder):
            if f.lower().endswith('.dxf'):
                files.append(f)

    return jsonify({'files': files})

@app.route('/download/<path:subfolder>/<path:filename>', methods=['GET'])
def download(subfolder, filename):
    """
    Download from a subfolder inside dxf_output
    """
    dxf_output_dir = os.path.join(os.getcwd(), 'uploads', 'dxf_output', subfolder)
    return send_from_directory(dxf_output_dir, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
