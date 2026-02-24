"""
Production-level PDF Processing Pipeline
========================================

This pipeline combines multiple components to process PDFs:
1. PDF → Images → Classification (good/bad pages)
2. Good pages → New PDF
3. PDF → MinerU processing → JSON extraction
4. JSON → Cerebras LLM → Structured output

Author: User (patched)
Date: October 2025
"""

import os
import sys
import json
import shutil
import subprocess
import time
import logging
import pathlib
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import gc
import re
from tensorflow.keras import backend as K
from groq import Groq
from pathlib import Path

# Third-party imports
import fitz  # PyMuPDF
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PDFProcessingPipeline:
    """Main pipeline class that orchestrates the entire PDF processing workflow."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration parameters.

        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.model = None
        self.cerebras_client = None
        self._setup()
        
        # self.groq_client = None # Add this line

    def _setup(self):
        """Setup the pipeline components and validate configuration."""
        logger.info("Setting up PDF Processing Pipeline...")

        # Validate required config
        required_keys = ['model_path', 'threshold', 'img_size', 'work_dir', 'gpu_device']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Create work directories
        self.work_dir = pathlib.Path(self.config['work_dir'])
        self.work_dir.mkdir(exist_ok=True)

        self.temp_dir = self.work_dir / "temp"
        self.output_dir = self.work_dir / "output"
        self.final_dir = self.work_dir / "final_results"

        for dir_path in [self.temp_dir, self.output_dir, self.final_dir]:
            dir_path.mkdir(exist_ok=True)

        # Load environment variables
        load_dotenv()

        # Initialize Cerebras client
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            logger.warning("CEREBRAS_API_KEY not found. Cerebras processing will be skipped.")
        else:
            self.cerebras_client = Cerebras(api_key=api_key)
            
        # # Initialize Groq client
        # groq_api_key = os.environ.get("GROQ_API_KEY")
        # if not groq_api_key:
        #     logger.warning("GROQ_API_KEY not found. Groq processing will be skipped.")
        # else:
        #     self.groq_client = Groq(api_key=groq_api_key)

        logger.info("Pipeline setup completed successfully.")

    def load_classification_model(self) -> keras.Model:
        """Load the trained CNN classification model."""
        if self.model is None:
            model_path = self.config['model_path']
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            logger.info(f"Loading classification model from {model_path}")
            self.model = keras.models.load_model(model_path)
            logger.info("Model loaded successfully")

        return self.model

    def preprocess_image(self, img_path: str) -> np.ndarray:
        """Preprocess image for model prediction."""
        target_size = tuple(self.config['img_size'])
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        return x

    def classify_image(self, img_path: str) -> Tuple[str, float, float]:
        """Classify a single image as good or bad."""
        model = self.load_classification_model()
        x = self.preprocess_image(img_path)

        prob = float(model.predict(x, verbose=0)[0][0])
        threshold = self.config['threshold']

        pred_class = "good" if prob >= threshold else "bad"
        confidence = prob if pred_class == "good" else (1.0 - prob)

        return pred_class, float(confidence), float(prob)

    def pdf_to_images_with_classification(self, pdf_path: str) -> Dict[str, Any]:
        """
        Convert PDF to images and classify each page.

        Args:
            pdf_path: Path to input PDF file

        Returns:
            Dictionary with classification results and paths
        """
        logger.info(f"Processing PDF: {pdf_path}")

        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_name = pathlib.Path(pdf_path).stem
        pdf_temp_dir = self.temp_dir / pdf_name
        good_dir = pdf_temp_dir / "good_pages"
        bad_dir = pdf_temp_dir / "bad_pages"

        # Create directories
        for dir_path in [pdf_temp_dir, good_dir, bad_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Open PDF and process pages
        doc = fitz.open(pdf_path)
        num_pages = len(doc)  # capture number of pages BEFORE any close
        zoom_x, zoom_y = self.config.get('zoom', (2.0, 2.0))
        rotation = self.config.get('rotation', 0)
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotation)

        good_pages = []
        bad_pages = []

        logger.info(f"Processing {num_pages} pages from PDF '{pdf_name}'")

        try:
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                image_name = f"page_{page_num + 1}.png"
                temp_image_path = pdf_temp_dir / image_name

                # Save page as image
                pix.save(str(temp_image_path))

                # Classify the page
                try:
                    pred_class, confidence, prob = self.classify_image(str(temp_image_path))
                except Exception as e:
                    logger.error(f"Error classifying page {page_num+1}: {e}")
                    pred_class, confidence, prob = "bad", 0.0, 0.0

                logger.info(f"Page {page_num + 1}: {pred_class} (confidence: {confidence:.3f}, prob: {prob:.4f})")

                # Destination path
                if pred_class == "good":
                    dest_path = good_dir / image_name
                    good_pages.append({
                        'page_num': page_num + 1,
                        'image_path': str(dest_path),
                        'confidence': confidence,
                        'prob': prob
                    })
                else:
                    dest_path = bad_dir / image_name
                    bad_pages.append({
                        'page_num': page_num + 1,
                        'image_path': str(dest_path),
                        'confidence': confidence,
                        'prob': prob
                    })

                # Move to appropriate directory (overwrite if exists)
                if dest_path.exists():
                    try:
                        os.remove(str(dest_path))
                    except Exception:
                        # fallback to replace
                        pass
                os.replace(str(temp_image_path), str(dest_path))

        finally:
            # ensure doc is always closed
            try:
                doc.close()
            except Exception:
                pass

        result = {
            'pdf_name': pdf_name,
            'total_pages': num_pages,
            'good_pages': good_pages,
            'bad_pages': bad_pages,
            'good_dir': str(good_dir),
            'bad_dir': str(bad_dir),
            'temp_dir': str(pdf_temp_dir)
        }

        logger.info(f"Classification complete: {len(good_pages)} good pages, {len(bad_pages)} bad pages")
        return result

    def create_pdf_from_good_pages(self, classification_result: Dict[str, Any], original_pdf_path: str) -> str:
        """
        Create a new PDF containing only good pages by copying them from the
        original document. This preserves vector content and keeps file size small.

        Args:
            classification_result: Result from pdf_to_images_with_classification.
            original_pdf_path: Path to the original PDF file.

        Returns:
            Path to the newly created PDF.
        """
        logger.info("Creating vector-preserving PDF from good pages...")

        good_pages_info = classification_result['good_pages']
        if not good_pages_info:
            raise ValueError("No good pages found to create PDF")
            
        # Get 0-based page indices from the 1-based page numbers
        good_page_indices = [p['page_num'] - 1 for p in good_pages_info]
        
        pdf_name = classification_result['pdf_name']
        new_pdf_path = self.temp_dir / f"{pdf_name}_good_pages.pdf"
        
        src_doc = None
        new_doc = None
        try:
            src_doc = fitz.open(original_pdf_path)
            new_doc = fitz.open()

            for i in good_page_indices:
                new_doc.insert_pdf(src_doc, from_page=i, to_page=i)
            
            # Save with options to reduce file size
            new_doc.save(str(new_pdf_path), garbage=4, deflate=True)
            
            logger.info(f"Created vector-preserving PDF: {new_pdf_path} with {len(good_page_indices)} pages")
            return str(new_pdf_path)
        
        except Exception as e:
            logger.error(f"Failed to create vector PDF: {e}")
            raise
        
        finally:
            if src_doc:
                src_doc.close()
            if new_doc:
                new_doc.close()

    def run_mineru_processing(self, pdf_path: str) -> Dict[str, Any]:
        """
        Run MinerU processing on the PDF to extract content.
        Args:
            pdf_path: Path to PDF file to process
        Returns:
            Dictionary with MinerU processing results
        """
        logger.info(f"Running MinerU processing on: {pdf_path}")

        pdf_name = pathlib.Path(pdf_path).stem
        mineru_output_dir = self.output_dir / "mineru_output"
        mineru_output_dir.mkdir(parents=True, exist_ok=True)

        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.config['gpu_device'])

        # MinerU command
        cmd = [
            "uv", "run", "mineru",
            "-p", pdf_path,
            "-o", str(mineru_output_dir),
            "-b", "vlm-vllm-engine",     
            "-l", "en",
            "-d", "cuda:0"
        ]

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info(f"Executing MinerU command: {' '.join(cmd)}")

        start_time = time.time()
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
            elapsed_time = time.time() - start_time

            if proc.returncode == 0:
                logger.info(f"MinerU processing completed successfully in {elapsed_time:.2f} seconds")

                content_list_path = None
                for root, _, files in os.walk(mineru_output_dir):
                    for file in files:
                        # Match either *_content_list.json
                        if file.endswith("_content_list.json"):
                            content_list_path = os.path.join(root, file)
                            break
                    if content_list_path:
                        break

                # If still not found, try recursive pattern match using pathlib
                if not content_list_path:
                    matches = list(pathlib.Path(mineru_output_dir).rglob("*_content_list.json"))
                    if matches:
                        content_list_path = str(matches[0])

                # --- Handle result ---
                if content_list_path and os.path.exists(content_list_path):
                    logger.info(f"Content list saved to: {content_list_path}")

                    return {
                        'success': True,
                        'content_list_path': str(content_list_path),
                        'processing_time': elapsed_time,
                        'mineru_output_dir': str(mineru_output_dir)
                    }
                else:
                    logger.error("Could not find content_list.json file in MinerU output")
                    return {'success': False, 'error': 'Content list file not found', 'stderr': proc.stderr}

            else:
                logger.error(f"MinerU processing failed: {proc.stderr}")
                return {'success': False, 'error': proc.stderr}

        except subprocess.TimeoutExpired:
            logger.error("MinerU processing timed out after 1 hour")
            return {'success': False, 'error': 'Processing timeout'}
        except Exception as e:
            logger.error(f"Error running MinerU: {str(e)}")
            return {'success': False, 'error': str(e)}

    def process_with_cerebras(self, content_json_path: str, pdf_name: str) -> Dict[str, Any]:
        """
        Process the extracted content with Cerebras LLM.
        Args:
            content_json_path: Direct path to the JSON file generated by MinerU
            pdf_name: Name of the original PDF (for naming the output file)
        """
        if not self.cerebras_client:
            logger.warning("Cerebras client not initialized. Skipping Cerebras processing.")
            return {'success': False, 'error': 'Cerebras API key not available'}

        logger.info(f"Processing content with Cerebras LLM using file: {content_json_path}")

        try:
            # Check if the path exists
            json_path_obj = pathlib.Path(content_json_path)
            if not json_path_obj.exists():
                raise FileNotFoundError(f"MinerU content JSON not found at: {content_json_path}")

            # Load MinerU JSON content
            with open(json_path_obj, 'r', encoding='utf-8') as f:
                content_data = f.read()

            # Prepare the extraction prompt (same logic, unchanged)
                prompt = """
    Use the set of rules:
    Extract structured information from this Bill of Lading into the given JSON schema.

    Schema Compliance Rules

    - Always return the full JSON schema exactly as provided.
    - Every field must be present.
    - If data is missing → set it to null.
    - Never skip or remove fields.

    Field-Specific Rules

    - "filingFor" → always "self".
    - "originalBlNumber" → always null.
    - "movemetType" → always null.
    - "typeOfBl" → only "Negotiable" or "Non-Negotiable" if explicitly written/stamped on the BL. Otherwise → null.
    - "natureOfCargo" → extract from “Description of Packages and Goods” (only what the cargo is, not counts or numbers).
    - "blDate" → always in dd-mm-yyyy format.
    - Numeric values (weights, volumes, counts) → plain numbers only (no commas, no units).

    HS Code Rules

    - Always extract any HS Code or Harmonized System Code found on the BL.
    - If multiple HS Codes appear → include all, separated by commas.
    - Apply the same HS Code(s) to every schema field that requires or references HS Code.
    - Never skip or assign it to only one field — ensure consistency across all related fields.

    Party Mapping Rules

    - "partyName" → always null.
    - Issuer determination:
    • By default → issuer = "Freight Forwarder".
    • Only if both “Forwarding Agent” and “Delivery Agent” are explicitly present → issuer = "Shipping Line Agent".

    - Determine the **issuing party type** as follows:

    • If the “issuing party” name or any word from the extracted data matches (exactly or partially) any of the shipping line names below → record it under "shippingLineAgent" fields.
    • Otherwise → record it under `"freightForwardingAgent"` fields.

    CONTAINER HANDLING RULES

    - Each container listed should appear as a separate sub-JSON in "containerDetails".
    Example:
    "containerDetails": [
    {
    "containerNumber": "MSCU1234567",
    "containerSize": "40HC",
    "containerStatus": "FCL",
    "shipperSeal": "S1234"
    },
    {
    "containerNumber": "MSCU9876543",
    "containerSize": "20GP",
    "containerStatus": "LCL",
    "shipperSeal": "S5678"
    }
    ]

    - "containerStatus" → one of ["FCL", "LCL", "EMPTY"] if written (case-insensitive); else null.
    - "containerSize" → extract “20'”, “40'”, “45'”, “GP”, “HC” etc.

    Seal Handling Rules

    - "shipperSeal" → always use the seal number (if provided).
    - "carrierSeal" → null unless explicitly listed separately.
    - Never use file numbers as seals.

    Special Markings / Remarks

    - Include any explicit notes or stamps found on the BL such as:
    • "TELEX RELEASE"
    • "EXPRESS RELEASE"
    • "Shipper's Load and Count"
    • "Said to Contain"
    - If multiple remarks → separate them with semicolons.


    Container Handling Rules
    - "containerStatus" → one of ["FCL", "LCL", "EMPTY"] as explicitly written in the text (case-insensitive).
    If none of these are mentioned → null.
    - "containerSize" → extract values like "20'", "40'", "45'", "GP", "HC", "HP" etc. (any such standard container size/type markings).
    If none → null.

    Location & Mapping Rules
    - Map fields using these equivalencies:
    • "portOfAcceptance" → from "Place of Receipt"
    • "portOfLoading" → from "Port of Loading"
    • "portOfUnloading" → from "Port of Discharge"
    • "endDestination" → from "Place of Delivery"
    - "placeOfIssue" → extract accurately from the BL (use nearby context or clearly labeled field such as "Place of Issue", "Issued at", or similar).

    General Guidance

    - Maintain clear text extraction (avoid OCR noise or formatting artifacts).
    - Only interpret what's explicitly stated; never infer.
    - Always ensure strict JSON structure compliance before returning the result.

    Extract the following fields from the given text in this JSON format:

    {{
    "blDetails": [
        {{
        "filingFor": "value",
        "partyName": "value",
        "partyMpciId": "value",
        "blNumber": "value",
        "blDate": "value",
        "parentBlNumber": "value",
        "parentBlIssuingPartyName": "value",
        "parentBlIssuingPartyMpciId": "value",
        "newSplitBlFiling": "value",
        "originalBlNumber": "value",
        "furtherConsolidation": "value",
        "consoleForwardersMpciId": "value",
        "vesselName": "value",
        "voyageNumber": "value",
        "movemetType": "value",
        "portOfAcceptance": "value",
        "portOfLoading": "value",
        "portOfTranshipment": "value",
        "portOfUnloading": "value",
        "endDestination": "value",
        "emirate": "value",
        "natureOfCargo": "value",
        "placeOfBillIssue": "value",
        "placeOfFreightPayment": "value",
        "typeOfBl": "value",
        "freightPrepaid": "value",
        "invoiceAmountOfTheCosginment": "value",
        "currency": "value",
        "serviceRequirement": "value",
        "remarks": "value",
        "shipperName": "value",
        "shipperTaxId": "value",
        "shipperAddress": "value",
        "shipperCity": "value",
        "shipperCountry": "value",
        "consigneeName": "value",
        "consigneeTaxID": "value",
        "consigneeAddress": "value",
        "consigneeCity": "value",
        "consigneeCountry": "value",
        "consigneeType": "value",
        "contactName": "value",
        "contactPartyType": "value",
        "phoneNumber": "value",
        "emailId": "value",
        "notifyPartyName": "value",
        "notifyPartyTaxID": "value",
        "notifyPartyAddress": "value",
        "notifyPartyCity": "value",
        "notifyPartyCountry": "value",
        "deliveryAgentName": "value",
        "deliveryAgentTaxID": "value",
        "deliveryAgentAddress": "value",
        "deliveryAgentCity": "value",
        "deliveryAgentCountry": "value",
        "freightForwardingAgentName": "value",
        "freightForwardingAgentTaxID": "value",
        "freightForwardingAgentAddress": "value",
        "freightForwardingAgentCity": "value",
        "freightForwardingAgentCountry": "value",
        "shippingLineAgentName": "value",
        "shippingLineAgentTaxID": "value",
        "shippingLineAgentAddress": "value",
        "shippingLineAgentCity": "value",
        "shippingLineAgentCountry": "value",
        "updateFilingReason": "value",
        "updateFilingRemarks": "value",
        "newSwitchBlNumber": "value"
        }}
    ],
    "containerDetails": [
        {{
        "blNumber": "value",
        "containerNumber": "value",
        "containerSize": "value",
        "containerStatus": "value",
        "shipperSeal": "value",
        "shipperSealType": "value",
        "carrierSeal": "value",
        "carrierSealType": "value",
        "customsSeal": "value",
        "customsSealType": "value",
        "sealingPartyName": "value",
        "temperature": "value",
        "temperatureUnit": "value",
        "numberOfPackages": "value",
        "packageType": "value",
        "vgmWeight": "value",
        "removeContainerForSplitFiling": "value",
        "newSplitBlNumber": "value",
        "newParentBlNumber": "value",
        "newVesselName": "value",
        "newVoyageNumber": "value"
        }}
    ],
    "itemDetails": [
        {{
        "blNumber": "value",
        "containerNumber": "value",
        "hsCode": "value",
        "numberOfPackages": "value",
        "packageType": "value",
        "packageRelatedDescriptionCode": "value",
        "cargoGrossWeight": "value",
        "cargoNetWeight": "value",
        "volumeInMtq": "value",
        "imoClass": "value",
        "unCode": "value",
        "itemDescription": "value",
        "marksAndNumbers": "value",
        "handlingInstructions": "value",
        "goodsItemCountryOfOrigin": "value",
        "removeItem": "value"
        }}
    ]
    }}

    Provide the output as JSON.
    """

            # Make Cerebras API call
            chat_completion = self.cerebras_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt + content_data
                    }
                ],
                model="gpt-oss-120b", # Ensure this model name is currently supported by your API tier
                stream=False,
                max_tokens=8000, # 50k might be too high for output tokens depending on model limits, adjusted to safe 8k
                temperature=0.5
            )

            response_content = chat_completion.choices[0].message.content

            # Save Cerebras output to final_results/
            response_path = self.final_dir / f"{pdf_name}_cerebras_response.json"

            # Try to parse and save the JSON output cleanly
            try:
                if '```json' in response_content:
                    json_start = response_content.find('```json') + 7
                    json_end = response_content.rfind('```')
                    json_str = response_content[json_start:json_end].strip()
                elif '```' in response_content: # Handle generic code blocks
                    json_start = response_content.find('```') + 3
                    json_end = response_content.rfind('```')
                    json_str = response_content[json_start:json_end].strip()
                else:
                    json_str = response_content.strip()

                parsed_response = json.loads(json_str)

                with open(response_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_response, f, indent=2, ensure_ascii=False)

            except json.JSONDecodeError:
                with open(response_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'raw_response': response_content,
                        'error': 'Could not parse response as JSON'
                    }, f, indent=2, ensure_ascii=False)

            logger.info(f"Cerebras response saved to: {response_path}")

            return {
                'success': True,
                'response_path': str(response_path),
                'response_content': response_content
            }

        except Exception as e:
            logger.error(f"Error processing with Cerebras: {str(e)}")
            return {'success': False, 'error': str(e)}


    # def process_with_groq(self, content_json_path: str) -> Dict[str, Any]:
    #     """
    #     Process the extracted content with the Groq LLM.

    #     Args:
    #         content_json_path: Path to the content_list.json file

    #     Returns:
    #         Dictionary with Groq processing results
    #     """
    #     if not self.groq_client:
    #         logger.warning("Groq client not initialized. Skipping Groq processing.")
    #         return {'success': False, 'error': 'GROQ_API_KEY not available'}

    #     logger.info("Processing content with Groq LLM...")

    #     try:
    #         # Load the content JSON
    #         with open(content_json_path, 'r', encoding='utf-8') as f:
    #             content_data = f.read()

    #         # The prompt remains the same
    #         prompt = f"""
    # Use the set of rules:
    # Extract structured information from this Bill of Lading into the given JSON schema.

    # Schema Compliance Rules
    # - Always return the full JSON schema exactly as provided.
    # - Every field must be present.
    # - If data is missing → set it to null.
    # - Never skip or remove fields.
    # Field-Specific Rules
    # - "filingFor" → always "self".
    # - "originalBlNumber" → always null.
    # - "movemetType" → always null.
    # - "typeOfBl" → only "Negotiable" or "Non-Negotiable" if explicitly written/stamped on the BL. Otherwise → null.
    # - "natureOfCargo" → extract from “Description of Packages and Goods” (only what the cargo is, not counts or numbers).
    # - "blDate" → always in dd-mm-yyyy format.
    # - Numeric values (weights, volumes, counts) → plain numbers only (no commas, no units).
    # HS Code Rules
    # - Always extract any HS Code or Harmonized System Code found on the BL.
    # - If multiple HS Codes appear → include all, separated by commas.
    # - Apply the same HS Code(s) to every schema field that requires or references HS Code.
    # - Never skip or assign it to only one field — ensure consistency across all related fields.
    # Party Mapping Rules
    # - "partyName" → always null.
    # - Issuer determination:
    # • By default → issuer = "Freight Forwarder".
    # • Only if both “Forwarding Agent” and “Delivery Agent” are explicitly present → issuer = "Shipping Line Agent".
    # • Never infer issuer from “As Carrier” or signatures — strictly follow above logic.
    # Seal Handling Rules
    # - "shipperSeal" → always use the seal number (if provided).
    # - "carrierSeal" → null unless explicitly listed separately.
    # - Never use file numbers as seals.
    # Special Markings / Remarks
    # - Include any explicit notes or stamps found on the BL such as:
    # • "TELEX RELEASE"
    # • "EXPRESS RELEASE"
    # • "Shipper's Load and Count"
    # • "Said to Contain"
    # - If multiple remarks → separate them with semicolons.
    # Container Handling Rules
    # - "containerStatus" → one of ["FCL", "LCL", "EMPTY"] as explicitly written in the text (case-insensitive).
    # If none of these are mentioned → null.
    # - "containerSize" → extract values like "20'", "40'", "45'", "GP", "HC", "HP" etc. (any such standard container size/type markings).
    # If none → null.
    # Location & Mapping Rules
    # - Map fields using these equivalencies:
    # • "portOfAcceptance" → from "Place of Receipt"
    # • "portOfLoading" → from "Port of Loading"
    # • "portOfUnloading" → from "Port of Discharge"
    # • "endDestination" → from "Place of Delivery"
    # - "placeOfIssue" → extract accurately from the BL (use nearby context or clearly labeled field such as "Place of Issue", "Issued at", or similar).
    # General Guidance
    # - Maintain clear text extraction (avoid OCR noise or formatting artifacts).
    # - Only interpret what's explicitly stated; never infer.
    # - Always ensure strict JSON structure compliance before returning the result.

    # Extract the following fields from the given text in this JSON format:
    # {{
    #   "blDetails": [
    #     {{
    #       "filingFor": "value",
    #       "partyName": "value",
    #       "partyMpciId": "value",
    #       "blNumber": "value",
    #       "blDate": "value",
    #       "parentBlNumber": "value",
    #       "parentBlIssuingPartyName": "value",
    #       "parentBlIssuingPartyMpciId": "value",
    #       "newSplitBlFiling": "value",
    #       "originalBlNumber": "value",
    #       "furtherConsolidation": "value",
    #       "consoleForwardersMpciId": "value",
    #       "vesselName": "value",
    #       "voyageNumber": "value",
    #       "movemetType": "value",
    #       "portOfAcceptance": "value",
    #       "portOfLoading": "value",
    #       "portOfTranshipment": "value",
    #       "portOfUnloading": "value",
    #       "endDestination": "value",
    #       "emirate": "value",
    #       "natureOfCargo": "value",
    #       "placeOfBillIssue": "value",
    #       "placeOfFreightPayment": "value",
    #       "typeOfBl": "value",
    #       "freightPrepaid": "value",
    #       "invoiceAmountOfTheCosginment": "value",
    #       "currency": "value",
    #       "serviceRequirement": "value",
    #       "remarks": "value",
    #       "shipperName": "value",
    #       "shipperTaxId": "value",
    #       "shipperAddress": "value",
    #       "shipperCity": "value",
    #       "shipperCountry": "value",
    #       "consigneeName": "value",
    #       "consigneeTaxID": "value",
    #       "consigneeAddress": "value",
    #       "consigneeCity": "value",
    #       "consigneeCountry": "value",
    #       "consigneeType": "value",
    #       "contactName": "value",
    #       "contactPartyType": "value",
    #       "phoneNumber": "value",
    #       "emailId": "value",
    #       "notifyPartyName": "value",
    #       "notifyPartyTaxID": "value",
    #       "notifyPartyAddress": "value",
    #       "notifyPartyCity": "value",
    #       "notifyPartyCountry": "value",
    #       "deliveryAgentName": "value",
    #       "deliveryAgentTaxID": "value",
    #       "deliveryAgentAddress": "value",
    #       "deliveryAgentCity": "value",
    #       "deliveryAgentCountry": "value",
    #       "freightForwardingAgentName": "value",
    #       "freightForwardingAgentTaxID": "value",
    #       "freightForwardingAgentAddress": "value",
    #       "freightForwardingAgentCity": "value",
    #       "freightForwardingAgentCountry": "value",
    #       "shippingLineAgentName": "value",
    #       "shippingLineAgentTaxID": "value",
    #       "shippingLineAgentAddress": "value",
    #       "shippingLineAgentCity": "value",
    #       "shippingLineAgentCountry": "value",
    #       "updateFilingReason": "value",
    #       "updateFilingRemarks": "value",
    #       "newSwitchBlNumber": "value"
    #     }}
    #   ],
    #   "containerDetails": [
    #     {{
    #       "blNumber": "value",
    #       "containerNumber": "value",
    #       "containerSize": "value",
    #       "containerStatus": "value",
    #       "shipperSeal": "value",
    #       "shipperSealType": "value",
    #       "carrierSeal": "value",
    #       "carrierSealType": "value",
    #       "customsSeal": "value",
    #       "customsSealType": "value",
    #       "sealingPartyName": "value",
    #       "temperature": "value",
    #       "temperatureUnit": "value",
    #       "numberOfPackages": "value",
    #       "packageType": "value",
    #       "vgmWeight": "value",
    #       "removeContainerForSplitFiling": "value",
    #       "newSplitBlNumber": "value",
    #       "newParentBlNumber": "value",
    #       "newVesselName": "value",
    #       "newVoyageNumber": "value"
    #     }}
    #   ],
    #   "itemDetails": [
    #     {{
    #       "blNumber": "value",
    #       "containerNumber": "value",
    #       "hsCode": "value",
    #       "numberOfPackages": "value",
    #       "packageType": "value",
    #       "packageRelatedDescriptionCode": "value",
    #       "cargoGrossWeight": "value",
    #       "cargoNetWeight": "value",
    #       "volumeInMtq": "value",
    #       "imoClass": "value",
    #       "unCode": "value",
    #       "itemDescription": "value",
    #       "marksAndNumbers": "value",
    #       "handlingInstructions": "value",
    #       "goodsItemCountryOfOrigin": "value",
    #       "removeItem": "value"
    #     }}
    #   ]
    # }}

    # Provide the output as JSON.
    # """ # Note: JSON schema from your prompt is truncated here for brevity

    #         # Make API call to Groq
    #         chat_completion = self.groq_client.chat.completions.create(
    #             messages=[
    #                 {
    #                     "role": "user",
    #                     "content": prompt + content_data
    #                 }
    #             ],
    #             model="openai/gpt-oss-120b",
    #             temperature=0.1,  # Lower temperature for more deterministic JSON output
    #             max_tokens=50000,
    #             response_format={"type": "json_object"}, # Enforce JSON output
    #         )

    #         response_content = chat_completion.choices[0].message.content

    #         # Save the response
    #         pdf_name = pathlib.Path(content_json_path).stem.replace('_content_list', '')
    #         response_path = self.final_dir / f"{pdf_name}_groq_response.json"

    #         # The response from Groq with response_format="json_object" should be a clean JSON string
    #         try:
    #             parsed_response = json.loads(response_content)
    #             with open(response_path, 'w', encoding='utf-8') as f:
    #                 json.dump(parsed_response, f, indent=2, ensure_ascii=False)
    #         except json.JSONDecodeError:
    #             # Fallback in case the response isn't perfect JSON
    #             with open(response_path, 'w', encoding='utf-8') as f:
    #                 json.dump({
    #                     'raw_response': response_content,
    #                     'error': 'Could not parse response as JSON'
    #                 }, f, indent=2, ensure_ascii=False)

    #         logger.info(f"Groq response saved to: {response_path}")

    #         return {
    #             'success': True,
    #             'response_path': str(response_path),
    #             'response_content': response_content
    #         }

    #     except Exception as e:
    #         logger.error(f"Error processing with Groq: {str(e)}")
    #         return {'success': False, 'error': str(e)}




    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF through the complete pipeline.

        Args:
            pdf_path: Path to the PDF file to process

        Returns:
            Dictionary with complete processing results
        """
        logger.info(f"Starting complete pipeline processing for: {pdf_path}")
        start_time = time.time()

        try:
            # Step 1: PDF to images with classification
            classification_result = self.pdf_to_images_with_classification(pdf_path)

            # Release TensorFlow model from GPU memory to free it for MinerU
            logger.info("Releasing TensorFlow model from memory...")
            del self.model
            self.model = None
            K.clear_session()
            gc.collect()
            logger.info("TensorFlow model released.")
        
            # Step 2: Create new PDF from good pages
            if classification_result['good_pages']:
                good_pdf_path = self.create_pdf_from_good_pages(classification_result, pdf_path)
            else:
                logger.warning("No good pages found. Using original PDF.")
                good_pdf_path = pdf_path

            # Step 3: Run MinerU processing
            mineru_result = self.run_mineru_processing(good_pdf_path)

            #if mineru_result.get('success'):
             #   cerebras_result = self.process_with_cerebras(mineru_result['content_list.json'])
                       
            # Step 4: Process with Cerebras (if available)
            cerebras_result = None
            if mineru_result.get('success'):
                # PDF stem for the output filename
                pdf_stem = pathlib.Path(pdf_path).stem
                
                # Get the actual path detected by run_mineru_processing
                detected_json_path = mineru_result.get('content_list_path')
                
                if detected_json_path:
                    # Pass the detected path AND the original pdf name
                    cerebras_result = self.process_with_cerebras(detected_json_path, pdf_stem)
                else:
                    logger.error("MinerU succeeded but content_list_path is missing")
                    cerebras_result = {'success': False, 'error': 'MinerU path missing'}

            #  # Step 4: Process with Groq (if available)
            # groq_result = None # Rename variable
            # if mineru_result.get('success'):
            #     groq_result = self.process_with_groq(mineru_result['content_list_path']) # Call new function

            # # Compile final results
            # # ...
            # final_result = {
            #     # ...
            #     'mineru_result': mineru_result,
            #     'groq_result': groq_result, # Rename key
            #     'success': mineru_result.get('success', False)
            # }
            
            # Compile final results
            total_time = time.time() - start_time

            final_result = {
                'pdf_path': pdf_path,
                'processing_time': total_time,
                'classification_result': classification_result,
                'good_pdf_path': good_pdf_path if classification_result['good_pages'] else None,
                'mineru_result': mineru_result,
                'cerebras_result': cerebras_result,
                'success': mineru_result.get('success', False)
            }

            # Save processing summary
            pdf_name = pathlib.Path(pdf_path).stem
            summary_path = self.final_dir / f"{pdf_name}_processing_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Pipeline processing completed in {total_time:.2f} seconds")
            logger.info(f"Processing summary saved to: {summary_path}")

            return final_result

        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            return {
                'pdf_path': pdf_path,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple PDFs through the pipeline.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            List of processing results for each PDF
        """
        logger.info(f"Starting batch processing of {len(pdf_paths)} PDFs")

        results = []
        for i, pdf_path in enumerate(pdf_paths, 1):
            logger.info(f"Processing PDF {i}/{len(pdf_paths)}: {pdf_path}")
            result = self.process_single_pdf(pdf_path)
            results.append(result)

        # Save batch summary
        batch_summary = {
            'total_pdfs': len(pdf_paths),
            'successful': sum(1 for r in results if r.get('success', False)),
            'failed': sum(1 for r in results if not r.get('success', False)),
            'results': results,
            'batch_processing_time': sum(r.get('processing_time', 0) for r in results)
        }

        batch_summary_path = self.final_dir / f"batch_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Batch processing completed. Summary saved to: {batch_summary_path}")
        logger.info(f"Results: {batch_summary['successful']} successful, {batch_summary['failed']} failed")

        return results

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the pipeline."""
    return {
        'model_path': 'best_transfer_model.h5',
        'threshold': 0.278,
        'img_size': [224, 224],
        'zoom': [2.0, 2.0],
        'rotation': 0,
        'work_dir': 'pipeline_work',
        'gpu_device': 1
    }

def enrich_shipping_line_in_json(final_json_path, mineru_text, shipping_line_list):
    """
    Finds a shipping line name in the mineru text and updates
    'shippingLineAgentName' in the JSON if found. Saves a new enriched JSON.
    """
    # Read the existing final JSON
    with open(final_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize text for case-insensitive search
    text_upper = mineru_text.upper()

    found_line = None
    for line in shipping_line_list:
        # Create a regex pattern for whole word match (case insensitive)
        pattern = r'\b' + re.escape(line.upper()) + r'\b'
        if re.search(pattern, text_upper):
            found_line = line
            break

    if found_line:
        print(f"Found shipping line: {found_line}")
        # Update only shippingLineAgentName field in blDetails
        if "blDetails" in data and isinstance(data["blDetails"], list):
            for bl in data["blDetails"]:
                if "shippingLineAgentName" in bl:
                    bl["shippingLineAgentName"] = found_line

        # Save updated JSON as *_final_output.json
        folder = os.path.dirname(final_json_path)
        filename = os.path.basename(final_json_path)
        new_filename = os.path.splitext(filename)[0] + "_final_output.json"
        new_path = os.path.join(folder, new_filename)

        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved enriched JSON to: {new_path}")
        return new_path
    else:
        print("No valid shipping line found.")
        return None


def main():
    """Main entry point for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='PDF Processing Pipeline')
    parser.add_argument('pdf_paths', nargs='+', help='Path(s) to PDF file(s) to process')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--model-path', default='best_transfer_model.h5', help='Path to trained model file')
    parser.add_argument('--threshold', type=float, default=0.278, help='Classification threshold')
    parser.add_argument('--work-dir', default='pipeline_work', help='Working directory for temporary files')
    parser.add_argument('--gpu-device', type=int, default=1, help='GPU device ID for MinerU processing')

    args = parser.parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()

    # Override config with command line arguments
    config.update({
        'model_path': args.model_path,
        'threshold': args.threshold,
        'work_dir': args.work_dir,
        'gpu_device': args.gpu_device
    })

    # Initialize and run pipeline
    pipeline = PDFProcessingPipeline(config)

    if len(args.pdf_paths) == 1:
        pdf_path = args.pdf_paths[0]
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        result = pipeline.process_single_pdf(pdf_path)
        print(f"Processing completed. Success: {result.get('success', False)}")
    else:
        results = pipeline.process_multiple_pdfs(args.pdf_paths)
        successful = sum(1 for r in results if r.get('success', False))
        print(f"Batch processing completed. {successful}/{len(results)} PDFs processed successfully.")
        # For batch mode, skip enrichment to avoid ambiguity on which file to use
        return

    # Shipping line list (unchanged)
    shipping_lines = [
        "ACL","Alianca","Allalouf","ANL","APL","Arkas","Atlanticargo","Avana Logistek",
        "Balaji Shipping","Beacon Intermodal","BLPL Singapore","Blue Sky","BMC Line",
        "Bridgehead","Brointermed lines","CAI","Canadian National Railway",
        "Canadian Pacific Railway","Carlisle Leasing","CARU","CCNI",
        "China Navigation Company","CHINA SHIPPING","CMA CGM","CNC Line","Combi Line",
        "Concor","Containerships Group","COSCO","Cosiarma","Cronos","Crowley",
        "CSAV NORASIA","CS Leasing","CULines","DAL","DB Schenker","Delmas","DFDS LYS",
        "DHL","Dolphin Line","Dong Fang","Dongjin Shipping","Dongnama",
        "Dong Young Shipping","DP World Nhava Sheva","Econ Shipping","ECU Worldwide",
        "Emirates Shipping Line","Emkay Line","ESL","EVERGREEN","FESCO","Finnlines",
        "Flexi-Van","Florens container services","Focus Trucking","GCL","GESeaCo",
        "Global Container International","GOLD","Gold Star Line",
        "Gothong Southern Shipping","Grimaldi","GVC","Haian","Hamburg Sud",
        "Hanjin Shipping","HAPAG LLOYD","Heung-A Shipping","Hyde Logistics","Hyundai",
        "Interasia Lines","Interport","IRISL","ITALIA MARITTIMA","Kambara Kisen",
        "KiwiRail","K-LINE","Korea Marine Transport","Kuehne + Nagel","Leschaco",
        "Libra","Linea Messina","MacAndrews","Macnels","MACS","MAERSK","MAG shipping",
        "Maldives Ports","Marfret","Marguisa","Mariana Express Lines","Matson",
        "Maxicon Container Line","Mediterranean Shipping Company","Medkon Lines",
        "Meratus Line","Minsheng Ocean Shipping","MOL","MSC","Namsung","NileDutch",
        "Nirint Shipping","NSCSA","NYK Line","Ocean Axis","Ocean World Lines",
        "ONE Line","OOCL","OTAL","Pacific Direct Line","Pan Asia Line",
        "Pan Continental Shipping","Pasha Hawaii","Perma Shipping Line","PIL",
        "Pollux & Castor","PSL Navegacao","Qatar Navigation","RCL","Roberto Bucci",
        "Royal Arctic","Safmarine","Saigon Newport Corporation","Samskip","Samudera",
        "Sarjak Container Lines","SCI","Seaboard Marine","SeaCastle Inc","Sea Hawk",
        "Sealand","Sea Star Line","SENATOR LINES","SETH Shipping",
        "Sinokor Merchant Marine","Sinotrans","SITC","SM Line","Sofrana ANL",
        "Sofrana Unilines","SPIL","Star Shipping","St. John Freight Systems",
        "Stolt Tank Containers","STX Pan Ocean","Sunmarine","TAL International",
        "Tanto","Tarros","Textainer","TIS-logistic","TOTE Maritime","Touax",
        "Trailer Bridge","Trans Asia","TransContainer","Transmar",
        "Transvision Shipping","Triton Container","T.S. Lines","Turkon","UES",
        "United Arab Shipping","VL Logistic","Wan Hai Lines","Waterfront Container",
        "W.E.C. Lines","Westwood","White Line","World Transport Overseas",
        "WorldWide Alliance","XPO Logistics","YANG MING (YML)","ZIM",
        "Zim World Freight","SeaLead Shipping","Sea Legend Line",
        "X-Press Feeders Shipping","MTT Shipping","Dongyoung Shipping",
        "SALAM PACIFIC INDONESIA LINES","Akkon Lines",
        "K Line / Kawasaki Kisen Kaisha","Dole Ocean Cargo Express","Boluda",
        "PT Temas Shipping","Ningbo Ocean Shipping Company","Indus Container Lines",
        "Transfar Shipping Pte. Ltd.","Independent Container Lines",
        "Hellmann Worldwide Logistics","Tanto Intim Line","JAS Worldwide",
        "Cheng Lie Navigation","Cargo Partner","King Ocean Services",
        "X-Press Container Line","Hecny Group","Hub Shipping","CK Line",
        "Atlantic Container Line","Trans Asian Shipping Services","EIMSKIP",
        "Orient Overseas Container Line","North Sea Container Line","Seatrade",
        "Milaha","Medkon Lines","Grimaldi Deep Sea S.P.A.","Namsung Shipping",
        "White Line Shipping","M-Line","NewStar","Transit LLC","Grimaldi Group",
        "BNSF Logistics","KAWA Shipping","Emkayline Shipping","PSL Navegacao",
        "Maxicon Container Line","Maersk Line Limited","Sea Legend Line",
        "Transworld Group Singapore","Lancer Container Lines",
        "Arkas Container Transport","Maersk Logistics and Services Ltd",
        "SETH Shipping","Route Planner","World Direct Shipping",
        "Deutsche Afrika-Linien","TransContainer","Unifeeder Group",
        "Crane Worldwide Logistics","Hai An Transport and Stevedoring",
        "Ethiopian Shipping Line","Nirint Shipping B.V","Ignazio Messina & C.",
        "Laurel Navigation","HR Lines","Kintetsu World Express","Boluda Line",
        "Eukor","Tangshan Port Hede Shipping","Bengal Tiger Lines","Meratus",
        "Swire Shipping","Carribean Feeder Services",
        "Far East State Shipping Company","TOTE Maritime","Gold Star Line",
        "CStar Line","OVP Shipping","Nauka Lines","Volta Container Line",
        "SAMSKIP","Tropical Shipping","Dsv Ocean Transport","Uni Global Shipping",
        "Dongjin Shipping","Altun Logistics","Cosiarma S.p.A.",
        "Kawasaki Kisen Kaisha","DHL Global Forwarding","PAN OCEAN",
        "Hamburg Süd","Lexzau Scharbau GmbH & Co. KG",
        "Maritime Carrier Shipping GmbH & Co","Wallenius Wilhelmsen",
        "Blue Water Lines","Blue Anchor America Line",
        "Asyad Shipping Compagny","Cordelia Container Shipping Line",
        "Vasco Maritime","Pan Asia Line","TURKON LINE","Tailwind Shipping Co.",
        "Sinokor Merchant Marine","Aladin Express","Asean Seas Line",
        "Nippon Yusen Kaisha","Great White Fleet","G Ocean","Hoegh Autoliners",
        "Geodis Group","Ellerman City Liners","Noatum Maritime",
        "Mariana Express Lines","Taicang Container Lines","Sealead Shipping",
        "Sea Hawk Lines","Messina Line","SITC Container Lines",
        "National Shipping of America","Korea Marine Transport Compagny",
        "Sealand","Samudera Shipping Line","Containerships",
        "Regional Container Lines","Pan Continental Shipping Co. Ltd.",
        "ECU Worldwide","CROWLEY Maritime","Polynesia Line",
        "Mitsui O.S.K. Lines","BAL Container Line","Hyundai Merchant Marine",
        "Amass Global Network Group Co.","Tarros","Camellia Line","Matson",
        "UWL","Marguisa Shipping Lines","Blue World Line",
        "Pasha Hawaii Transport","DongYoung Shipping","Maritime Marfret",
        "Sunmarine Shipping Services","Akkon Lines",
        "Neptune Pacific Direct Line","Avana Global FZCO","Nippon Express",
        "Shanghai Jinjiang Shipping","NVO Consolidation B.V","Yusen Logistics",
        "Alianca Navegacao e Logistica","Nile Dutch Africa Line","Dachser",
        "Hillebrand Gori","Perma Shipping Line","Minsheng Ocean Shipping",
        "PT Temas Shipping","Transvision Shipping Line",
        "Odyssey Logistics & Technology","Admiral Container Lines",
        "Sarjak Container Lines","BMC Line Shipping","Stolt Tank Containers",
        "Zhonggu Logistics Corps","American President Lines",
        "Shipping Corporation of India","Vanguard Logistics","Romocean",
        "Safetrans Line","Peel Ports Group","Ningbo Ocean Shipping Co.",
        "Emirates Shipping Line","Bee Logistics",
        "West European Container Lines","Reel Shipping","T.S. Lines",
        "MTT Shipping","Allalouf Shipping Line","Econship",
        "ANL Container Line","Kambara Kisen","Orient Star Group",
        "Sinotrans Container Lines","Global Feeder Shipping",
        "Carpenters Shipping","Seino Logix Co","Heung-A Shipping",
        "COSCO Shipping Specialized Carriers Co., Ltd.","China United Lines",
        "KMTC"
    ]

    # Safely derive paths from the actual result instead of hardcoding
    mineru_result = result.get('mineru_result', {}) if isinstance(result, dict) else {}
    mineru_text_path = mineru_result.get('content_list_path')

    cerebras_result = result.get('cerebras_result', {}) if isinstance(result, dict) else {}
    final_json_path = cerebras_result.get('response_path')

    if (
        mineru_text_path
        and final_json_path
        and os.path.exists(mineru_text_path)
        and os.path.exists(final_json_path)
    ):
        try:
            with open(mineru_text_path, 'r', encoding='utf-8') as f:
                mineru_text = f.read()

            enrich_shipping_line_in_json(final_json_path, mineru_text, shipping_lines)
        except Exception as e:
            print(f"Error during shipping line enrichment: {e}")
    else:
        print("Skipping shipping line enrichment: required JSON files not available.")


if __name__ == "__main__":
    main()