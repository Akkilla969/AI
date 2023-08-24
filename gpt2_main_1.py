{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Akkilla969/AI/blob/main/gpt2_main_1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "Zq6yIkkzM7_6",
        "outputId": "95d78777-3791-4369-8e19-2ef57b7c5800",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting transformers[torch]\n",
            "  Downloading transformers-4.32.0-py3-none-any.whl (7.5 MB)\n",
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/7.5 MB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.4/7.5 MB\u001b[0m \u001b[31m42.2 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m7.5/7.5 MB\u001b[0m \u001b[31m124.2 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m7.5/7.5 MB\u001b[0m \u001b[31m82.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (3.12.2)\n",
            "Collecting huggingface-hub<1.0,>=0.15.1 (from transformers[torch])\n",
            "  Downloading huggingface_hub-0.16.4-py3-none-any.whl (268 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m268.8/268.8 kB\u001b[0m \u001b[31m16.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (1.23.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (23.1)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (6.0.1)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (2023.6.3)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (2.31.0)\n",
            "Collecting tokenizers!=0.11.3,<0.14,>=0.11.1 (from transformers[torch])\n",
            "  Downloading tokenizers-0.13.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.8 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m7.8/7.8 MB\u001b[0m \u001b[31m80.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting safetensors>=0.3.1 (from transformers[torch])\n",
            "  Downloading safetensors-0.3.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.3/1.3 MB\u001b[0m \u001b[31m58.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (4.66.1)\n",
            "Requirement already satisfied: torch!=1.12.0,>=1.9 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (2.0.1+cu118)\n",
            "Collecting accelerate>=0.20.3 (from transformers[torch])\n",
            "  Downloading accelerate-0.22.0-py3-none-any.whl (251 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m251.2/251.2 kB\u001b[0m \u001b[31m23.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate>=0.20.3->transformers[torch]) (5.9.5)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.15.1->transformers[torch]) (2023.6.0)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.15.1->transformers[torch]) (4.7.1)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.9->transformers[torch]) (1.12)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.9->transformers[torch]) (3.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.9->transformers[torch]) (3.1.2)\n",
            "Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.9->transformers[torch]) (2.0.0)\n",
            "Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch!=1.12.0,>=1.9->transformers[torch]) (3.27.2)\n",
            "Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch!=1.12.0,>=1.9->transformers[torch]) (16.0.6)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers[torch]) (3.2.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers[torch]) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers[torch]) (2.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers[torch]) (2023.7.22)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch!=1.12.0,>=1.9->transformers[torch]) (2.1.3)\n",
            "Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch!=1.12.0,>=1.9->transformers[torch]) (1.3.0)\n",
            "Installing collected packages: tokenizers, safetensors, huggingface-hub, transformers, accelerate\n",
            "Successfully installed accelerate-0.22.0 huggingface-hub-0.16.4 safetensors-0.3.3 tokenizers-0.13.3 transformers-4.32.0\n"
          ]
        }
      ],
      "source": [
        "!pip install transformers[torch]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "XKHQoX4eNGYc",
        "outputId": "2b952776-16f4-4a47-bf97-10d4f0435be9",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: accelerate in /usr/local/lib/python3.10/dist-packages (0.22.0)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from accelerate) (1.23.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (23.1)\n",
            "Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate) (5.9.5)\n",
            "Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from accelerate) (6.0.1)\n",
            "Requirement already satisfied: torch>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (2.0.1+cu118)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.12.2)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (4.7.1)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (1.12)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.1.2)\n",
            "Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2.0.0)\n",
            "Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.10.0->accelerate) (3.27.2)\n",
            "Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.10.0->accelerate) (16.0.6)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.10.0->accelerate) (2.1.3)\n",
            "Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.10.0->accelerate) (1.3.0)\n"
          ]
        }
      ],
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
          "base_uri": "https://localhost:8080/",
          "height": 231,
          "referenced_widgets": [
            "1715feea62b848149914cef52b511ac8",
            "a15c050a5c8d46e9a223d6ecec82d3af",
            "58b067f741b544ceb666e95db3f5f532",
            "09a923a0890a4d44b40259ac3a1c1e33",
            "e6024f4235984b9c8e8c7ed8eaa4c84e",
            "da4d427919644a9bae37289f821f01ea",
            "e0d88f39b5674531ad20d4f0f672214e",
            "84428d82eb1f47edb346e1a800837e52",
            "a012ee998950435baed9e4885e723fbe",
            "12584feeb96e4031ba8496acffec40f7",
            "e4886a35286949ef958fd57be4ae90a6",
            "27314eb59ec24677a087c25539ee6122",
            "ce6140400790497285e273cff3dc2747",
            "4782793a17e645f6b835033244e3de05",
            "b7a8e41cceb548f38cfa976ccb82c598"
          ]
        },
        "id": "XWeoro7EPINm",
        "outputId": "d773e00c-76c1-4f6e-c2e7-7895e1222e6b"
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
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "Downloading (…)neration_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "e6024f4235984b9c8e8c7ed8eaa4c84e"
            }
          },
          "metadata": {}
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
              "      <progress value='348' max='2750' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
              "      [ 348/2750 38:50 < 4:29:40, 0.15 it/s, Epoch 6.31/50]\n",
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
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bIZb_5rza2H9",
        "outputId": "e72de271-0b81-4697-a28a-18301a594138"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.\n",
            "Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "\n",
            "\n",
            "\n",
            " Medicines and related substances act as a \"drug importation and export control authority\" for the purposes of this Act. The Medicines and Related Substances Act (42 U.S.C. 1395 et seq.) is amended by adding at the end the following new section: ``(e) Regulations.-- ``(1) Regulations issued under this section shall specify the terms and conditions for the importation and export of the Medicines and Related Substances Act (42 U.S.C. 13\n"
          ]
        }
      ],
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
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RM1tiJiNeK0t",
        "outputId": "a01fddf9-4aa3-4819-ab63-9088d712268a"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
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
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yKJcVtPweLl_",
        "outputId": "3963d7f9-efd4-45f2-e1a9-f113f39764f3"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "('/content/drive/MyDrive/gpt2_finetuned/tokenizer_config.json',\n",
              " '/content/drive/MyDrive/gpt2_finetuned/special_tokens_map.json',\n",
              " '/content/drive/MyDrive/gpt2_finetuned/vocab.json',\n",
              " '/content/drive/MyDrive/gpt2_finetuned/merges.txt',\n",
              " '/content/drive/MyDrive/gpt2_finetuned/added_tokens.json')"
            ]
          },
          "execution_count": 13,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
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
    },
    "widgets": {
      "application/vnd.jupyter.widget-state+json": {
        "e6024f4235984b9c8e8c7ed8eaa4c84e": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_da4d427919644a9bae37289f821f01ea",
              "IPY_MODEL_e0d88f39b5674531ad20d4f0f672214e",
              "IPY_MODEL_84428d82eb1f47edb346e1a800837e52"
            ],
            "layout": "IPY_MODEL_a012ee998950435baed9e4885e723fbe"
          }
        },
        "da4d427919644a9bae37289f821f01ea": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_12584feeb96e4031ba8496acffec40f7",
            "placeholder": "​",
            "style": "IPY_MODEL_e4886a35286949ef958fd57be4ae90a6",
            "value": "Downloading (…)neration_config.json: 100%"
          }
        },
        "e0d88f39b5674531ad20d4f0f672214e": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_27314eb59ec24677a087c25539ee6122",
            "max": 124,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_ce6140400790497285e273cff3dc2747",
            "value": 124
          }
        },
        "84428d82eb1f47edb346e1a800837e52": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_4782793a17e645f6b835033244e3de05",
            "placeholder": "​",
            "style": "IPY_MODEL_b7a8e41cceb548f38cfa976ccb82c598",
            "value": " 124/124 [00:00&lt;00:00, 6.04kB/s]"
          }
        },
        "a012ee998950435baed9e4885e723fbe": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "12584feeb96e4031ba8496acffec40f7": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "e4886a35286949ef958fd57be4ae90a6": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "27314eb59ec24677a087c25539ee6122": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "ce6140400790497285e273cff3dc2747": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "4782793a17e645f6b835033244e3de05": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "b7a8e41cceb548f38cfa976ccb82c598": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        }
      }
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}