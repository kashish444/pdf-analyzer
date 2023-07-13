# PDF Analyzer

PDF Analyzer is a Python application that allows you to upload a PDF file, extract its text content, and perform question-answering on the extracted text using OpenAI's language models.

## Prerequisites

Before running the application, ensure that you have the following prerequisites installed:

- Python (version 3.11.3)
- pip (package installer for Python)
- An OpenAI API key 

## Execution Steps

Follow these steps to run the PDF Analyzer application:

1. Install the required dependencies using pip:

2. Set up the environment variables:
- Create a file named ".env" in the project root directory.
- Add the following line to the ".env" file, replacing "your_openai_api_key" with your OpenAI API key:
  ```
  openai_api_key=your_openai_api_key
  ```

3. Run the application:

4. The application will open in your browser. Use the sidebar to upload a PDF file or you can also drag and drop your PDF file.

5. Once the PDF is uploaded, you can ask questions about its content in the provided text input field.

6. The application will use OpenAI's language model to perform a similarity search on the PDF's text content and generate a response to your question.

## Dependencies

The PDF Analyzer application has the following dependencies:

- os
- pickle
- streamlit
- dotenv
- PyPDF2
- langchain
- streamlit_extras
- OpenAI GPT-3.5



