{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Akkilla969/AI/blob/main/gpt2_main.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Zq6yIkkzM7_6"
      },
      "outputs": [],
      "source": [
        "!pip install transformers[torch]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "XKHQoX4eNGYc"
      },
      "outputs": [],
      "source": [
        "!pip install accelerate -U"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TfRm0R5zc_LP"
      },
      "source": [
        "text-inference pairs might be better suited\n",
        "\n",
        "Has not been trained with temperature value"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/",
          "height": 275,
          "referenced_widgets": [
            "1715feea62b848149914cef52b511ac8",
            "a15c050a5c8d46e9a223d6ecec82d3af",
            "58b067f741b544ceb666e95db3f5f532",
            "09a923a0890a4d44b40259ac3a1c1e33",
            "e6024f4235984b9c8e8c7ed8eaa4c84e"
          ]
        },
        "id": "XWeoro7EPINm",
        "outputId": "af4603b4-9381-4aae-eb02-a34282f0c20b"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "1715feea62b848149914cef52b511ac8",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading (…)olve/main/vocab.json:   0%|          | 0.00/1.04M [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "a15c050a5c8d46e9a223d6ecec82d3af",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading (…)olve/main/merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "58b067f741b544ceb666e95db3f5f532",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading (…)lve/main/config.json:   0%|          | 0.00/665 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/transformers/data/datasets/language_modeling.py:53: FutureWarning: This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "09a923a0890a4d44b40259ac3a1c1e33",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading model.safetensors:   0%|          | 0.00/548M [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "e6024f4235984b9c8e8c7ed8eaa4c84e",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading (…)neration_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "    <div>\n",
              "      \n",
              "      <progress value='251' max='2750' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
              "      [ 251/2750 27:55 < 4:40:14, 0.15 it/s, Epoch 4.55/50]\n",
              "    </div>\n",
              "    <table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              " <tr style=\"text-align: left;\">\n",
              "      <th>Step</th>\n",
              "      <th>Training Loss</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "  </tbody>\n",
              "</table><p>"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "    <div>\n",
              "      \n",
              "      <progress value='277' max='2750' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
              "      [ 277/2750 30:50 < 4:37:23, 0.15 it/s, Epoch 5.02/50]\n",
              "    </div>\n",
              "    <table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              " <tr style=\"text-align: left;\">\n",
              "      <th>Step</th>\n",
              "      <th>Training Loss</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "  </tbody>\n",
              "</table><p>"
            ]
          },
          "metadata": {}
        }
      ],
      "source": [
        "import torch\n",
        "from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments\n",
        "\n",
        "# Load the GPT-2 tokenizer\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")\n",
        "\n",
        "# Load and preprocess your text dataset\n",
        "dataset = TextDataset(\n",
        "    tokenizer=tokenizer,\n",
        "    file_path=\"/content/pdf_data_4.json\",  # Path to your text dataset file\n",
        "    block_size=128  # Specify the desired maximum sequence length\n",
        ")\n",
        "\n",
        "# Create a data collator for language modeling\n",
        "data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)\n",
        "\n",
        "# Initialize the GPT-2 model\n",
        "model = GPT2LMHeadModel.from_pretrained(\"gpt2\")\n",
        "\n",
        "# Set up the training arguments\n",
        "training_args = TrainingArguments(\n",
        "    output_dir=\"./output_dir\",  # Directory to save the trained model\n",
        "    overwrite_output_dir=True,\n",
        "    num_train_epochs=50,  # Number of training epochs\n",
        "    per_device_train_batch_size=4,\n",
        "    save_steps=500,  # Save model checkpoints every 500 steps\n",
        "    save_total_limit=2 # Save only the last 2 checkpoints\n",
        "\n",
        ")\n",
        "\n",
        "# Create a Trainer instance for training\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    data_collator=data_collator,\n",
        "    train_dataset=dataset\n",
        ")\n",
        "\n",
        "# Train the model\n",
        "trainer.train()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "YLcbqSNDbdlj"
      },
      "outputs": [],
      "source": [
        "model.save_pretrained(\"/content/finetuned_model\")\n",
        "tokenizer.save_pretrained(\"/content/finetuned_model\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TfKlWjZIb5hw"
      },
      "outputs": [],
      "source": [
        "model = GPT2LMHeadModel.from_pretrained(\"/content/finetuned_model\")\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(\"/content/finetuned_model\")\n",
        "\n",
        "# Function to generate an answer given a question\n",
        "def generate_answer(question):\n",
        "    input_ids = tokenizer.encode(question, return_tensors=\"pt\")\n",
        "    output = model.generate(input_ids, max_length=100)\n",
        "    answer = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "    return answer\n",
        "\n",
        "# Ask a question and get an answer\n",
        "prompt = \"an inference made is that: \"\n",
        "answer = generate_answer(prompt)\n",
        "\n",
        "# prompt = \"Medicines and related substances act\"\n",
        "# answer = generate_answer(prompt)\n",
        "\n",
        "print(\"\\n\\n\\n\\n\",answer)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bIZb_5rza2H9"
      },
      "outputs": [],
      "source": [
        "model = GPT2LMHeadModel.from_pretrained(\"/content/finetuned_model\")\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(\"/content/finetuned_model\")\n",
        "\n",
        "# Function to generate an answer given a question\n",
        "def generate_answer(question):\n",
        "    input_ids = tokenizer.encode(question, return_tensors=\"pt\")\n",
        "    output = model.generate(input_ids, max_length=100)\n",
        "    answer = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "    return answer\n",
        "\n",
        "# Ask a question and get an answer\n",
        "# prompt = \"Inference from this text is that\"\n",
        "# answer = generate_answer(prompt)\n",
        "\n",
        "prompt = \"Medicines and related substances act\"\n",
        "answer = generate_answer(prompt)\n",
        "\n",
        "print(\"\\n\\n\\n\\n\",answer)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RM1tiJiNeK0t"
      },
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "\n",
        "# Mount Google Drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yKJcVtPweLl_"
      },
      "outputs": [],
      "source": [
        "model.save_pretrained(\"/content/drive/MyDrive/gpt2_finetuned\")\n",
        "tokenizer.save_pretrained(\"/content/drive/MyDrive/gpt2_finetuned\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import requests\n",
        "from transformers import GPT2LMHeadModel, GPT2Tokenizer\n",
        "\n",
        "# Load the fine-tuned GPT-2 model and tokenizer\n",
        "model_path = \"path_to_fine_tuned_model\"\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(model_path)\n",
        "model = GPT2LMHeadModel.from_pretrained(model_path)\n",
        "\n",
        "# Fetch the CSV file\n",
        "def fetch_csv(file_path):\n",
        "    try:\n",
        "        data = pd.read_csv(file_path)\n",
        "        return data\n",
        "    except FileNotFoundError:\n",
        "        return None\n",
        "\n",
        "# Perform statistical analysis on the CSV data\n",
        "def get_statistical_analysis(data):\n",
        "    mode_value = df[column_name].mode().values[0]\n",
        "    # std_value = df[column_name].std()\n",
        "    # variance_value = df[column_name].var()\n",
        "    count_value = df[column_name].count()\n",
        "    max_value = df[column_name].max()\n",
        "    min_value = df[column_name].min()\n",
        "    mean = data['column_name'].mean()\n",
        "    return [mode_value,count_value,max_value,min_value,mean]\n",
        "\n",
        "# Fetch data from the internet\n",
        "def fetch_data_from_internet(keywords):\n",
        "    response = requests.get('https://www.{keywords}.com/')\n",
        "    data = response.json()\n",
        "    return data\n",
        "\n",
        "def summarize_article(article):\n",
        "    # Tokenize and encode the input article\n",
        "    inputs = tokenizer.encode(article, return_tensors='pt')\n",
        "\n",
        "    # Generate the summary using the GPT-2 model\n",
        "    summary_ids = model.generate(inputs, max_length=100, num_return_sequences=1, early_stopping=True)\n",
        "    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)\n",
        "\n",
        "    return summary\n",
        "\n",
        "# Chatbot functionality\n",
        "def chatbot():\n",
        "    while True:\n",
        "        user_input = input(\"User: \")\n",
        "        if user_input == \"quit\":\n",
        "            break\n",
        "        csv_data = fetch_csv(\"path_to_csv_file.csv\")\n",
        "        if csv_data is None:\n",
        "            # CSV data not found, fetch from the internet\n",
        "            csv_data = fetch_data_from_internet()\n",
        "\n",
        "        if csv_data is None:\n",
        "            print(\"Data not available. Please try again later.\")\n",
        "            continue\n",
        "\n",
        "        statistical_insight = get_statistical_analysis(csv_data)\n",
        "\n",
        "        if user_input.lower().startswith(\"summary\"):\n",
        "            # User requested article summarization\n",
        "            article = user_input[7:]\n",
        "            # Generate the article summary\n",
        "            summary = summarize_article(article)\n",
        "            print(\"Summary:\", summary)\n",
        "        else:\n",
        "            # User asked a question\n",
        "            print(\"Statistical analysis result:\", statistical_insight)\n",
        "chatbot()\n"
      ],
      "metadata": {
        "id": "GStovAriKOWN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "http://192.168.1.9:8502"
      ],
      "metadata": {
        "id": "uRcgJqaTMmSG"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
