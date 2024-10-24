# Hotel Chatbot

Welcome to the Hotel Chatbot repository! This chatbot is designed to assist users with hotel-related inquiries and tasks. Below are instructions on how to set up and run the chatbot.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Running the Chatbot

Once the dependencies are installed, you can run the chatbot using Streamlit. Execute the following command:

```bash
streamlit run app.py
```

This will start the chatbot interface, allowing users to interact with it.

## Project Structure

- **app.py**: This file contains the Streamlit application code for running the chatbot interface.

- **src/model.py**: The `model.py` file contains the implementation of the neural network model used for the chatbot.

- **src/utils.py**: The `utils.py` file provides utilities such as tokenizer, stemmer, and bag of words used for preprocessing text data.

- **training/train.py**: The `train.py` file is responsible for training the chatbot model.

- **training/evaluation.py**: The `evaluation.py` file contains code for evaluating the performance of the trained model.

- **artifacts/intents.json**: This JSON file contains all the intents that the model has been trained on. Intents represent different types of user queries or interactions.

- **artifacts/Hotel_model.pth**: The `Hotel_model.pth` file stores the details of the trained chatbot model.

- **artifacts/chat_history.json**: The `chat_history.json` file is used to save the chat history, including past user queries and bot responses.

## Usage

The Hotel Chatbot is capable of answering various hotel-related queries such as room availability, amenities, local attractions, and more. Users can interact with the chatbot by typing their questions or messages into the chat interface.

## Contributors

- AJIN T

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and use it according to your needs.

If you have any questions or issues, please feel free to contact me through ajinravi04@gamil.com . Thank you for using the Hotel Chatbot!