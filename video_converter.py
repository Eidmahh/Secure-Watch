#!/usr/bin/env python3

import cv2
import os
import tempfile
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class VideoConverter:
    
    @staticmethod
    def get_video_codec(video_path: str) -> Optional[str]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            fourcc_str = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            cap.release()
            
            return fourcc_str.strip()
        except Exception as e:
            logger.error(f"Error getting video codec: {e}")
            return None
    
    @staticmethod
    def is_browser_compatible(video_path: str) -> bool:
        codec = VideoConverter.get_video_codec(video_path)
        if not codec:
            return False
        
        # Browser-compatible codecs
        compatible_codecs = ['H264', 'h264', 'avc1', 'X264', 'AVC1']
        return codec in compatible_codecs
    
    @staticmethod
    def is_fmp4_video(video_path: str) -> bool:
        codec = VideoConverter.get_video_codec(video_path)
        return codec == 'FMP4' if codec else False
    
    @staticmethod
    def convert_to_h264(input_path: str, output_path: str = None) -> Optional[str]:
        try:
            # Create temporary output file if not specified
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                output_path = os.path.join(temp_dir, f"{base_name}_h264.mp4")
            
            logger.info(f"Converting {input_path} to H.264 format: {output_path}")
            
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"Could not open input video: {input_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Input video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Define H.264 codec - try different options for better compatibility
            # Try H.264 codecs in order of preference
            codec_options = [
                cv2.VideoWriter_fourcc(*'H264'),  # H.264 codec
                cv2.VideoWriter_fourcc(*'X264'),  # X264 codec
                cv2.VideoWriter_fourcc(*'avc1'),  # AVC1 codec
                cv2.VideoWriter_fourcc(*'mp4v'),  # MP4V fallback
            ]
            
            fourcc = None
            for codec in codec_options:
                test_out = cv2.VideoWriter(output_path, codec, fps, (width, height))
                if test_out.isOpened():
                    fourcc = codec
                    test_out.release()
                    break
                test_out.release()
            
            if fourcc is None:
                logger.error("No compatible codec found")
                cap.release()
                return None
            
            # Create output video writer
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error("Could not open output video writer")
                cap.release()
                return None
            
            # Convert frame by frame
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frame_count += 1
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(f"Conversion progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Release resources
            cap.release()
            out.release()
            
            # Verify output file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Conversion successful: {output_path} ({os.path.getsize(output_path)} bytes)")
                return output_path
            else:
                logger.error("Conversion failed: output file is empty or doesn't exist")
                return None
                
        except Exception as e:
            logger.error(f"Error during video conversion: {e}")
            return None
    
    @staticmethod
    def get_converted_video_path(original_path: str) -> str:
        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        return os.path.join(temp_dir, f"{base_name}_h264_converted.mp4")
    
    @staticmethod
    def cleanup_converted_video(converted_path: str):
        try:
            if os.path.exists(converted_path):
                os.remove(converted_path)
                logger.info(f"Cleaned up converted video: {converted_path}")
        except Exception as e:
            logger.warning(f"Could not clean up converted video {converted_path}: {e}")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample video
    test_video = "Video_Alert/fmp4_backup/Violence_Alert_Live Camera Feed_20250613_162334_781.mp4"
    
    if os.path.exists(test_video):
        print(f"Testing video: {test_video}")
        
        codec = VideoConverter.get_video_codec(test_video)
        print(f"Codec: {codec}")
        
        is_compatible = VideoConverter.is_browser_compatible(test_video)
        print(f"Browser compatible: {is_compatible}")
        
        is_fmp4 = VideoConverter.is_fmp4_video(test_video)
        print(f"Is FMP4: {is_fmp4}")
        
        if not is_compatible:
            print("Converting to H.264...")
            converted_path = VideoConverter.convert_to_h264(test_video)
            if converted_path:
                print(f"Conversion successful: {converted_path}")
                
                # Test the converted video
                converted_codec = VideoConverter.get_video_codec(converted_path)
                print(f"Converted codec: {converted_codec}")
                
                converted_compatible = VideoConverter.is_browser_compatible(converted_path)
                print(f"Converted video browser compatible: {converted_compatible}")
            else:
                print("Conversion failed")
    else:
        print(f"Test video not found: {test_video}") 