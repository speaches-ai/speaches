#!/usr/bin/env python3
"""
Interactive Speaker Diarization Example

This script demonstrates how to use pyannote.audio for speaker diarization.
It loads an audio file, identifies who spoke when, and allows mapping speakers
to custom names for easy understanding.

Usage:
    python src/diarization.py path/to/audio.wav
    
Requirements:
    pip install pyannote.audio
    
Note: You'll need to accept the user conditions and get an access token from
Hugging Face for pyannote models: https://huggingface.co/pyannote/speaker-diarization
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation
except ImportError:
    print("Error: pyannote.audio is not installed.")
    print("Please install it with: pip install pyannote.audio")
    exit(1)


class DiarizationProcessor:
    """
    A class to handle speaker diarization using pyannote.audio
    """
    
    def __init__(self, use_auth_token: Optional[str] = None):
        """
        Initialize the diarization processor
        
        Args:
            use_auth_token: Hugging Face authentication token (optional)
        """
        self.use_auth_token = use_auth_token
        self.pipeline = None
        self.speaker_mapping: Dict[str, str] = {}
    
    def load_model(self):
        """
        Load the pyannote speaker diarization pipeline
        """
        print("Loading pyannote speaker diarization model...")
        try:
            # Load the pre-trained pipeline
            # Note: You need to accept user conditions on HuggingFace for this model
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.use_auth_token
            )
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTroubleshooting tips:")
            print("1. Accept user conditions at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("2. Get an access token from: https://huggingface.co/settings/tokens")
            print("3. Set environment variable: export HUGGINGFACE_TOKEN='your_token'")
            print("4. Or pass token as argument: --token your_token")
            raise
    
    def perform_diarization(self, audio_path: str) -> Annotation:
        """
        Perform speaker diarization on an audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Annotation object containing diarization results
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Processing audio file: {audio_path}")
        
        # Perform diarization
        diarization = self.pipeline(audio_path)
        
        print(f"✓ Diarization complete! Found {len(diarization.labels())} speakers.")
        return diarization
    
    def setup_speaker_mapping(self, diarization: Annotation) -> None:
        """
        Interactive setup to map speaker labels to custom names
        
        Args:
            diarization: The diarization annotation results
        """
        speakers = sorted(diarization.labels())
        
        print("\n" + "="*50)
        print("SPEAKER MAPPING SETUP")
        print("="*50)
        print("Detected speakers:", speakers)
        print("\nLet's assign meaningful names to each speaker.")
        print("(Press Enter to use default names like 'Speaker_0', 'Speaker_1', etc.)")
        
        for speaker in speakers:
            while True:
                custom_name = input(f"\nEnter name for {speaker} (or press Enter for default): ").strip()
                
                if not custom_name:
                    # Use default naming
                    default_name = f"Speaker_{speaker.split('_')[-1] if '_' in speaker else speaker}"
                    self.speaker_mapping[speaker] = default_name
                    break
                elif custom_name not in self.speaker_mapping.values():
                    # Use custom name if it's unique
                    self.speaker_mapping[speaker] = custom_name
                    break
                else:
                    print(f"Name '{custom_name}' is already used. Please choose a different name.")
        
        print("\n✓ Speaker mapping complete!")
        for original, mapped in self.speaker_mapping.items():
            print(f"  {original} → {mapped}")
    
    def print_diarization_results(self, diarization: Annotation) -> None:
        """
        Print detailed diarization results with timestamps
        
        Args:
            diarization: The diarization annotation results
        """
        print("\n" + "="*60)
        print("SPEAKER DIARIZATION RESULTS")
        print("="*60)
        
        # Sort segments by start time
        segments = sorted(diarization.itertracks(yield_label=True), 
                         key=lambda x: x[0].start)
        
        print(f"{'Start Time':<12} {'End Time':<12} {'Duration':<10} {'Speaker':<15}")
        print("-" * 60)
        
        total_duration = 0
        speaker_durations = {}
        
        for segment, _, speaker in segments:
            start_time = segment.start
            end_time = segment.end
            duration = end_time - start_time
            
            # Get mapped speaker name
            speaker_name = self.speaker_mapping.get(speaker, speaker)
            
            # Format time as MM:SS
            start_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
            end_str = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
            duration_str = f"{duration:.1f}s"
            
            print(f"{start_str:<12} {end_str:<12} {duration_str:<10} {speaker_name:<15}")
            
            # Track statistics
            total_duration += duration
            if speaker_name not in speaker_durations:
                speaker_durations[speaker_name] = 0
            speaker_durations[speaker_name] += duration
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total analyzed duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print("\nSpeaking time per person:")
        
        for speaker, duration in sorted(speaker_durations.items()):
            percentage = (duration / total_duration) * 100
            print(f"  {speaker:<15}: {duration:>6.1f}s ({percentage:>5.1f}%)")
    
    def save_results(self, diarization: Annotation, output_path: str) -> None:
        """
        Save diarization results to a text file
        
        Args:
            diarization: The diarization annotation results
            output_path: Path to save the results
        """
        with open(output_path, 'w') as f:
            f.write("Speaker Diarization Results\n")
            f.write("=" * 30 + "\n\n")
            
            segments = sorted(diarization.itertracks(yield_label=True), 
                             key=lambda x: x[0].start)
            
            for segment, _, speaker in segments:
                start_time = segment.start
                end_time = segment.end
                speaker_name = self.speaker_mapping.get(speaker, speaker)
                
                start_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                end_str = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
                
                f.write(f"{start_str} - {end_str}: {speaker_name}\n")
        
        print(f"\n✓ Results saved to: {output_path}")


def main():
    """
    Main function to run the interactive diarization example
    """
    parser = argparse.ArgumentParser(
        description="Interactive Speaker Diarization with pyannote.audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/diarization.py audio.wav
  python src/diarization.py meeting.mp3 --token your_hf_token
  python src/diarization.py interview.wav --output results.txt
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to process"
    )
    
    parser.add_argument(
        "--token",
        help="Hugging Face authentication token (can also use HUGGINGFACE_TOKEN env var)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file to save results (optional)"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive speaker naming"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file '{args.audio_file}' does not exist.")
        return 1
    
    # Get authentication token
    token = args.token or os.getenv("HUGGINGFACE_TOKEN")
    
    # Display banner
    print("\n" + "="*60)
    print("PYANNOTE.AUDIO SPEAKER DIARIZATION EXAMPLE")
    print("="*60)
    print(f"Audio file: {args.audio_file}")
    print(f"Using HF token: {'Yes' if token else 'No (may cause authentication issues)'}")
    
    try:
        # Initialize processor
        processor = DiarizationProcessor(use_auth_token=token)
        
        # Load the model
        processor.load_model()
        
        # Perform diarization
        diarization = processor.perform_diarization(args.audio_file)
        
        # Setup speaker mapping (unless disabled)
        if not args.no_interactive:
            processor.setup_speaker_mapping(diarization)
        else:
            # Use default mapping
            for speaker in diarization.labels():
                default_name = f"Speaker_{speaker.split('_')[-1] if '_' in speaker else speaker}"
                processor.speaker_mapping[speaker] = default_name
        
        # Print results
        processor.print_diarization_results(diarization)
        
        # Save results if requested
        if args.output:
            processor.save_results(diarization, args.output)
        
        print("\n✓ Diarization complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
