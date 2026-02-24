# PDF Processing Pipeline

A production-level pipeline that combines PDF processing, image classification, and document analysis using MinerU and Cerebras AI.

## Overview

This pipeline processes PDF documents through multiple stages:

1. **PDF to Images**: Converts PDF pages to images using PyMuPDF
2. **Page Classification**: Uses a trained CNN model to classify pages as "good" or "bad"
3. **PDF Reconstruction**: Creates a new PDF containing only the good pages
4. **Content Extraction**: Uses MinerU to extract structured content from the processed PDF
5. **AI Analysis**: Processes the extracted content using Cerebras AI for structured data extraction

## Features

- Production-ready code with comprehensive error handling
- Detailed logging and progress tracking
- Configurable parameters via JSON configuration
- Organized output structure with temporary file management
- Support for single PDF or batch processing
- Detailed processing reports and summaries
- Optimized for shipping document processing (Bills of Lading)

## Installation

### Prerequisites

1. **Python 3.8+** with virtual environment support
2. **CUDA-compatible GPU** for MinerU processing
3. **MinerU installation** (using uv package manager)
4. **Trained CNN model** file (`best_transfer_model.h5`)

### Setup

1. **Clone or download the pipeline files**
   ```bash
   # Ensure you have all pipeline files:
   # - pdf_processing_pipeline.py
   # - requirements.txt
   # - pipeline_config.json
   ```

2. **Create virtual environment**
   ```bash
   python -m venv pipeline_env
   source pipeline_env/bin/activate  # On Windows: pipeline_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install MinerU** (using uv package manager)
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install MinerU
   uv pip install mineru
   ```

5. **Setup environment variables**
   ```bash
   # Create .env file
   echo "CEREBRAS_API_KEY=your_cerebras_api_key_here" > .env
   ```

6. **Prepare model file**
   - Ensure `best_transfer_model.h5` is in the same directory as the pipeline script
   - Update the `model_path` in `pipeline_config.json` if the model is elsewhere

## Configuration

Edit `pipeline_config.json` to customize the pipeline:

```json
{
  "model_path": "best_transfer_model.h5",
  "threshold": 0.278,
  "img_size": [224, 224],
  "zoom": [2.0, 2.0],
  "rotation": 0,
  "work_dir": "pipeline_work",
  "gpu_device": 0,
  "batch_size": 1,
  "timeout_seconds": 3600,
  "cleanup_temp_files": true,
  "save_intermediate_results": true
}
```

### Configuration Parameters

- `model_path`: Path to the trained CNN classification model
- `threshold`: Classification threshold for good/bad page detection
- `img_size`: Input image size for the CNN model [width, height]
- `zoom`: Zoom factors for PDF to image conversion [x, y]
- `rotation`: Rotation angle for image conversion
- `work_dir`: Working directory for temporary and output files
- `gpu_device`: GPU device ID for MinerU processing
- `timeout_seconds`: Maximum processing time per PDF
- `cleanup_temp_files`: Whether to clean up temporary files after processing
- `save_intermediate_results`: Whether to save intermediate processing results

## Usage

### Single PDF Processing

```bash
python pdf_processing_pipeline.py input.pdf
```

### Batch Processing

```bash
python pdf_processing_pipeline.py file1.pdf file2.pdf file3.pdf
```

### Custom Configuration

```bash
python pdf_processing_pipeline.py input.pdf --config custom_config.json
```

### Command Line Options

```bash
python pdf_processing_pipeline.py --help
```

Available options:
- `--config`: Path to custom configuration JSON file
- `--model-path`: Path to trained model file
- `--threshold`: Classification threshold
- `--work-dir`: Working directory for temporary files
- `--gpu-device`: GPU device ID for processing

## Output Structure

The pipeline creates the following output structure:

```
pipeline_work/
├── temp/                          # Temporary files
│   └── pdf_name/
│       ├── good_pages/           # Classified good pages
│       ├── bad_pages/            # Classified bad pages
│       └── pdf_name_good_pages.pdf
├── output/
│   └── mineru_output/            # MinerU processing results
└── final_results/
    ├── pdf_name_content_list.json      # Extracted content from MinerU
    ├── pdf_name_cerebras_response.json # AI-processed structured data
    └── pdf_name_processing_summary.json # Complete processing report
```

## Output Files

### Content List JSON
Contains the raw extracted content from MinerU processing, including:
- Text content
- Table structures
- Image references
- Layout information

### Cerebras Response JSON
Contains structured data extracted by Cerebras AI, including:
- Document information (type, number, date)
- Parties (shipper, consignee, notify party)
- Cargo details (description, weight, volume)
- Shipping information (ports, vessel, container)
- Commercial terms (freight, payment)

### Processing Summary JSON
Contains complete processing metadata:
- Processing times for each stage
- Classification results
- File paths and locations
- Success/failure status
- Error messages (if any)

## Logging

The pipeline generates comprehensive logs:
- Console output for real-time monitoring
- `pipeline.log` file for detailed processing history
- Individual processing reports for each PDF

## Error Handling

The pipeline includes robust error handling:
- ✅ File validation and existence checks
- ✅ Model loading validation
- ✅ GPU availability verification
- ✅ MinerU processing timeout handling
- ✅ API rate limiting and retry logic
- ✅ Graceful degradation for missing components

## Performance Optimization

- **GPU Processing**: Utilizes CUDA for MinerU processing
- **Batch Processing**: Efficient handling of multiple PDFs
- **Memory Management**: Proper cleanup of temporary files
- **Concurrent Processing**: Can be extended for parallel processing

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure `best_transfer_model.h5` exists in the specified path
   - Check file permissions

2. **CUDA/GPU issues**
   - Verify CUDA installation: `nvidia-smi`
   - Check GPU device ID in configuration

3. **MinerU processing fails**
   - Ensure `uv` and `mineru` are properly installed
   - Check available GPU memory
   - Verify PDF file integrity

4. **Cerebras API errors**
   - Verify API key in `.env` file
   - Check internet connectivity
   - Monitor API rate limits

### Debug Mode

Enable detailed debugging by modifying the logging level in the pipeline:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Integration

The pipeline can be integrated into larger workflows:

```python
from pdf_processing_pipeline import PDFProcessingPipeline

# Initialize with custom config
config = {
    'model_path': 'path/to/model.h5',
    'work_dir': 'custom_work_dir',
    # ... other config options
}

pipeline = PDFProcessingPipeline(config)

# Process single PDF
result = pipeline.process_single_pdf('document.pdf')

# Process multiple PDFs
results = pipeline.process_multiple_pdfs(['doc1.pdf', 'doc2.pdf'])
```

## Requirements

### Hardware Requirements
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- 16GB+ RAM for large PDF processing
- SSD storage for better I/O performance

### Software Requirements
- Python 3.8+
- CUDA Toolkit 11.8+ (for GPU acceleration)
- TensorFlow 2.12+ (with CUDA support)
- MinerU (via uv package manager)

## License

This pipeline is designed for internal use and document processing workflows.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for detailed error messages
3. Verify all dependencies and configurations
4. Test with sample PDFs to isolate issues

---

**Note**: This pipeline is optimized for shipping document processing, particularly Bills of Lading, but can be adapted for other document types by modifying the Cerebras prompt and classification model.
