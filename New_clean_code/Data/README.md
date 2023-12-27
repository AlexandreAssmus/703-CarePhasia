# AphasiaBank Data Cleaning Process Documentation

## Summary
This README outlines the process for transforming and cleaning speech data from AphasiaBank's `.cha` files, part of the TalkBank project. The focus is on analyzing speech from individuals with aphasia, preserving unique speech features like hesitations and repetitions, while removing non-essential elements such as interviewer dialogues (INV) and additional pragmatic markings.

## 1. Processing AphasiaBank `.cha` Files
### Description
`.cha` files contain detailed transcriptions with speaker markers and metadata, used in AphasiaBank for speech studies focusing on aphasia.

### Process
- **Text Extraction:** Lines marked with `*PAR:` representing the participant's speech were extracted.
- **Conversion to JSON:** The extracted data was then converted into a JSON format for ease of processing and analysis.

## 2. Conversion from JSON to CSV
### Description
Converting the data to CSV format facilitates its manipulation and analysis using standard data processing tools.

### Process
- **Data Filtering:** Select and convert only relevant fields from JSON to CSV format.
- **Directory Structure:** The CSV files' directory structure mirrors the original `.cha` files.

## 3. Cleaning and Refinement of CSV Data
### Description
The aim is to obtain clean text that accurately reflects the participant's speech, free from non-essential elements for linguistic analysis.

### Process
- **Removal of INV Dialogues:** All lines corresponding to the interviewer's speech (INV) were removed.
- **Preservation of Sequenced Speech:** Natural speech, including hesitations and repetitions, was preserved to maintain discourse authenticity.
- **Elimination of Pragmatic Markings:** Removed unnecessary pragmatic markings and additional annotations.
- **Symbol and Number Cleaning:** Removed all unnecessary symbols and numbers, except for apostrophes.

## 4. Final Considerations
### Data Validation
It is crucial to validate the data quality at each stage to ensure suitability for linguistic studies in aphasia.

### Automation and Repeatability
The process is designed for automation and can be applied to new `.cha` data sets from AphasiaBank, facilitating ongoing research.

### Documentation and Code
The code used in each stage is fully documented for clarity and ease of adaptation. Includes detailed instructions for executing scripts and required system setup.

---

For any additional information or specific inquiries related to the data cleaning process, please feel free to reach out.


