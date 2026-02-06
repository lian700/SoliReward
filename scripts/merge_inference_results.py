#!/usr/bin/env python3
"""
Merge inference results from multiple GPUs into a single JSON file.
"""
import json
try:
    import orjson
except ImportError:
    orjson = None
import glob
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm


def merge_results(input_dir: str, pattern: str = "output_gpu_*.json", output_file: str = None):
    """
    Merge multiple JSON result files into one.
    
    Args:
        input_dir: Directory containing the result files
        pattern: File pattern to match (default: 'output_gpu_*.json')
        output_file: Path to merged output file (default: input_dir/merged_output.json)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        return False
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}", file=sys.stderr)
        return False
    
    # Build search pattern
    search_pattern = str(input_path / pattern)
    print(f"Searching for files matching: {search_pattern}")
    
    output_files = sorted(glob.glob(search_pattern))
    
    if not output_files:
        print(f"Error: No files found matching pattern: {search_pattern}", file=sys.stderr)
        return False
    
    print(f"Found {len(output_files)} files to merge")
    
    all_results = []
    failed_files = []
    
    for output_file_path in output_files:
        try:
            file_size = os.path.getsize(output_file_path)
            with open(output_file_path, 'r', encoding='utf-8') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Loading {Path(output_file_path).name}") as pbar:
                    content = ''
                    chunk_size = 8192
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        content += chunk
                        pbar.update(len(chunk.encode('utf-8')))
                    data = orjson.loads(content) if orjson else json.loads(content)
                all_results.extend(data)
                print(f"✓ Loaded {len(data)} samples from {Path(output_file_path).name}")
        except Exception as e:
            print(f"✗ Error loading {Path(output_file_path).name}: {e}", file=sys.stderr)
            failed_files.append(output_file_path)
    
    if not all_results:
        print("Error: No valid data loaded from any file", file=sys.stderr)
        return False
    
    # Determine output file path
    if output_file is None:
        output_file = str(input_path / "merged_output.json")
    
    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Write merged results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Merge completed successfully!")
    print(f"Total samples merged: {len(all_results)}")
    print(f"Output file: {output_file}")
    
    if failed_files:
        print(f"\nWarning: {len(failed_files)} files failed to load:")
        for failed_file in failed_files:
            print(f"  - {Path(failed_file).name}")
    
    print(f"{'='*60}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge inference results from multiple GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all output_gpu_*.json files in a directory
  python3 merge_inference_results.py --input_dir result-infer/multi_gpu_202510242252

  # Merge with custom pattern
  python3 merge_inference_results.py --input_dir result-infer/multi_gpu_202510242252 --pattern "output_*.json"

  # Specify custom output file
  python3 merge_inference_results.py --input_dir result-infer/multi_gpu_202510242252 --output_file final_results.json
        """
    )
    parser.add_argument("--input_dir", "-i", type=str, required=True, 
                        help="Directory containing the result files")
    parser.add_argument("--pattern", type=str, default="output_gpu_*.json",
                        help="File pattern to match (default: output_gpu_*.json)")
    parser.add_argument("--output_file", "-o", type=str, default=None,
                        help="Path to merged output file (default: input_dir/merged_output.json)")
    
    args = parser.parse_args()
    
    try:
        success = merge_results(args.input_dir, args.pattern, args.output_file)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
