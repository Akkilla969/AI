{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "C2EgqEPDQ8v6"
      },
      "source": [
        "## Finetune Llama-2-7b on a Google colab\n",
        "\n",
        "Welcome to this Google Colab notebook that shows how to fine-tune the recent Llama-2-7b model on a single Google colab and turn it into a chatbot\n",
        "\n",
        "We will leverage PEFT library from Hugging Face ecosystem, as well as QLoRA for more memory efficient finetuning"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "i-tTvEF1RT3y"
      },
      "source": [
        "## Setup\n",
        "\n",
        "Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes). We will also install `einops` as it is a requirement to load Falcon models."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "Ss-sOGu0Ts13"
      },
      "outputs": [],
      "source": [
        "import pandas as pd"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Only for a json file to load it up when rbi data is stored in form of json1.json file\n"
      ],
      "metadata": {
        "id": "8Vb8cBGXa_3b"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "AsWB7LJWTmyz"
      },
      "outputs": [],
      "source": [
        "# json_file_path = '/content/json1.json'\n",
        "\n",
        "# # Load the JSON data\n",
        "# json_data = load_json_file(json_file_path)\n",
        "\n",
        "# # Convert JSON data to a pandas DataFrame\n",
        "# df = pd.DataFrame(json_data)\n",
        "\n",
        "# # Convert the DataFrame to a datasets.Dataset\n",
        "# dataset = datasets.Dataset.from_pandas(df)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zH1XVFL_jhXe",
        "outputId": "571ea58f-f0f9-46b1-d68a-9026ea273359"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: datasets in /usr/local/lib/python3.10/dist-packages (2.14.4)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from datasets) (1.23.5)\n",
            "Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (9.0.0)\n",
            "Requirement already satisfied: dill<0.3.8,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.3.7)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (1.5.3)\n",
            "Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2.31.0)\n",
            "Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (4.66.1)\n",
            "Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets) (3.3.0)\n",
            "Requirement already satisfied: multiprocess in /usr/local/lib/python3.10/dist-packages (from datasets) (0.70.15)\n",
            "Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (2023.6.0)\n",
            "Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.8.5)\n",
            "Requirement already satisfied: huggingface-hub<1.0.0,>=0.14.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.16.4)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets) (23.1)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (6.0.1)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (23.1.0)\n",
            "Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (3.2.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.0.4)\n",
            "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)\n",
            "Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.9.2)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.4.0)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (3.12.2)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (4.7.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2023.7.22)\n",
            "Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2023.3)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install datasets\n",
        "import datasets\n",
        "from datasets import Dataset, DatasetDict\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Loading the dataset from jsonfile,converting it to dataset object"
      ],
      "metadata": {
        "id": "8Sa9wCJ9mp79"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import datasets\n",
        "\n",
        "json_data=[{'text': '###human: What is the reference number for the recruitment of Assistant Manager in IRDAI? ###assistant: Ref: irdai/hr/am/dr rectt./02/2023'}, {'text': '###human: When will the Phase-II Descriptive Examination be held for the Assistant Manager post in IRDAI? ###assistant: The Phase-II Descriptive Examination will be held on 05-08-2023.'}, {'text': '###human: How can candidates download the call letter for the Phase-II Descriptive Examination? ###assistant: Candidates can download the call letter for the Phase-II Descriptive Examination by clicking on the link provided below: Click here to download the call letter.'}, {'text': '###human: What other information can candidates download from the given links? ###assistant: Candidates can also download the Sure Handpalk - English / Information Handout - English, and Sure Handpoint Dastak - Raji / Information Handout - Hindi from the provided links.'}, {'text': '###human: How will the candidates be intimated about the examination details? ###assistant: Candidates will receive an intimation notification separately by email/SMS.'}, {'text': \"###human: What is the procedure for candidates to claim travel expenses for the examination? ###assistant: Candidates need to download the Travel Claim (TC) Form by clicking on the link provided. They must fill in the required details and take a printout of the filled TC form. The form has to be submitted along with relevant tickets for travel to and from the examination center. The eligible amount of reimbursement will be credited to the candidate's bank account as per the details furnished in the TC form.\"}, {'text': \"###human: How will the travel expenses be reimbursed for candidates called for the Phase II - Descriptive Examination? ###assistant: Candidates called for the Phase II - Descriptive Examination will be reimbursed for the actual Single II AC Railway Fare. The reimbursement will be based on the shortest route from the candidate's residence/place of work to the center of the examination.\"}, {'text': '###human: When will the Preliminary Examination for the Assistant Manager post in IRDAI be held? ###assistant: The Preliminary Examination will be held on 25-06-2023.'}, {'text': '###human: What other information can candidates download from the given links? ###assistant: Candidates can also download the Sure Handpalk - English / Information Handout - English, and Sure Handpoint Dastak - Raji / Information Handout - Hindi from the provided links.'}, {'text': '###human: How will the candidates be intimated about the examination results? ###assistant: Candidates will be intimated about the examination results through a separate email/SMS.'}, {'text': '###human: What is the reference number for the recruitment of Assistant Manager in IRDAI? ###assistant: Ref: IRDAI/HR/Am/Dr Rectt./02/2023/vol-209.06.2023'}, {'text': '###human: When will the Phase I Examination for the Assistant Manager post in IRDAI be held? ###assistant: The Phase I Examination will be held on 25-06-2023, starting from 08:30 AM.'}, {'text': '###human: What do candidates need to do for the Phase I Examination? ###assistant: Candidates need to undergo surprise registration, and the examination letter and information handouts will be posted soon. All registered candidates will take the examination at the allocated examination centers.'}, {'text': '###human: How can candidates download the call letter and information handouts for the Phase I Examination? ###assistant: The link to download the call letter and information handouts will be posted soon.'}, {'text': '###human: Where will the Phase I Examination take place? ###assistant: The Phase I Examination will take place at the allocated examination centers.'}, {'text': '###human: What is the purpose of the notification related to the examination? ###assistant: The notification is related to the examination of the nomination of Assistant Manager.'}, {'text': '###human: What is the duration of the Pre-Examination Training for the Assistant Manager post? ###assistant: The Pre-Examination Training will be conducted from 31-05-2023 to 03-06-2023.'}, {'text': '###human: Who will conduct the Pre-Examination Training? ###assistant: The Pre-Examination Training will be conducted online by the Institute of Insurance and Risk Management (IIRM).'}, {'text': '###human: Who is eligible for the Pre-Examination Training? ###assistant: Candidates belonging to sc/st/obc/pwbd categories and have sent their requests for training in terms of the IRDAI notification dated 11-04-2023 are eligible for the Pre-Examination Training.'}, {'text': '###human: How will the candidates receive the schedule and link for the Pre-Examination Training? ###assistant: IIRM will share the webex link and schedule of the training with the candidates who have opted for pre-examination training through emails.'}, {'text': '###human: What is the reference number for the Deputation Appointment of Officers? ###assistant: Ref: HR/Deputation/04/2023'}, {'text': '###human: What is the purpose of the Corrigendum notification? ###assistant: The Corrigendum notification is for the acceptance of applications on the subject of Appointment of Officers on Deputation.'}, {'text': '###human: What is the new last date for receipt of applications? ###assistant: The last date for receipt of applications has been extended from 05.05.2023 to 15.06.2023.'}, {'text': '###human: What is the date of the original notification? ###assistant: The original notification was dated 21.04.2023.'}, {'text': '###human: What are the existing terms and conditions for deputation? ###assistant: The existing terms and conditions of deputation remain unchanged as per the notification dated 21.04.2023.'}, {'text': '###human: What is the purpose of the provided text? ###assistant: The provided text is a form of Caste Certificate for SC/ST (Scheduled Caste/Scheduled Tribe) individuals, used for certifying their caste/tribe status.'}, {'text': '###human: What are the different orders under which the Scheduled Caste/Tribe status is recognized? ###assistant: The Scheduled Caste/Tribe status is   recognized under various orders, such as: The Constitution Scheduled Castes Order, 1950, The Constitution Scheduled Tribes Order, 1950, The Constitution (Scheduled Castes) (Union Territories) (Part C States) Order, 1951, And several other Acts and Orders mentioned in the provided text.'}, {'text': '###human: Who is authorized to issue Caste Certificates? ###assistant: Caste Certificates can be issued by the following authorities: District Magistrate/Collector/Deputy Commissioner or any designated officials in the district. Chief Presidency Magistrate/Additional Chief Presidency Magistrate/President Magistrate. Revenue Officer not below the rank of Tehsildar. Certificates issued by Gazetted Officers of the Central or State Government countersigned by the concerned District Magistrate. Sub-Divisional Officer of the area where the candidate and/or their family normally reside.'}, {'text': '###human: Are there any specific procedures to be followed while applying for the Caste Certificate? ###assistant: Yes, the specific procedures for applying for a Caste Certificate may vary from one state/union territory to another. Generally, the applicant needs to fill out an application form, provide necessary documents supporting their caste claim, and submit it to the designated authority in the area of residence. The authority will then process the application and issue the Caste Certificate if the eligibility criteria are met.'}, {'text': \"###human: What is the significance of the Caste Certificate? ###assistant: The Caste Certificate is significant as it provides official recognition of an individual's Scheduled Caste or Scheduled Tribe status. It helps in availing various benefits, reservations, and opportunities provided by the government for the welfare and upliftment of SC/ST communities.\"}, {'text': '###human: Can individuals from certain territories also apply for the Caste Certificate? ###assistant: Yes, individuals from specific territories, like Dadra and Nagar Haveli, Pondicherry, Goa, Daman and Diu, Nagaland, Sikkim, Andaman and Nicobar Islands, etc., can also apply for the Caste Certificate under their respective Scheduled Castes/Scheduled Tribes Orders.'}, {'text': '###human: How long is the Caste Certificate valid? ###assistant: Caste Certificates are usually valid indefinitely. However, in some cases, the certificate may need to be renewed if requested or deemed necessary by the concerned authorities.'}, {'text': '###human: What are the requirements for obtaining a Caste Certificate for SC/ST individuals? ###assistant: To obtain a Caste Certificate for SC/ST individuals, the following requirements must be met: The applicant must belong to a Scheduled Caste or Scheduled Tribe as recognized by the specific constitutional orders or Acts mentioned in the form. The certificate must be issued based on the Scheduled Caste/Scheduled Tribe certificate of either the father or the mother of the applicant. The certificate must be issued by authorized officers, such as District Magistrate/Collector/Deputy Commissioner, Chief Presidency Magistrate, Revenue Officer, or Gazetted Officers of the Central or State Government, as per the place of residence.'}, {'text': \"###human: Is there any specific procedure for obtaining certificate of disability? ###assistant: Yes, the procedure for obtaining this certificate may vary depending on the country or region's regulations. Generally, an individual seeking the Certificate of Disability needs to approach a notified medical authority, undergo a medical examination, and submit the necessary documents to prove their disability status. The medical authority will then issue the certificate if the person's condition meets the specified guidelines and criteria.\"}, {'text': '###human: Who is the medical authority responsible for issuing this certificate? ###assistant: The medical authority responsible for issuing this certificate is not specified in the provided text. However, in most cases, it is issued by a notified medical authority, often a qualified medical practitioner or a panel of medical experts designated by the government.'}, {'text': '###human: What are the specific disabilities that disability certificate covers? ###assistant: This certificate covers the following specific disabilities: locomotor disability, dwarfism, blindness.'}, {'text': '###human: What is the purpose of the provided disbaility certificate? ###assistant: The provided text is a template or form for issuing a Certificate of Disability. It is used to certify the disability status of an individual with specific conditions such as amputation or complete permanent paralysis of limbs, dwarfism, or blindness.'}, {'text': '###human: What information does the Certificate of Disability contain? ###assistant: The Certificate of Disability contains the following information: Name, address, and date of birth of the person with a disability.Type of disability (e.g., locomotor disability, dwarfism, or blindness). Percentage of permanent disability in relation to a specific part of the body (if applicable). Registration number and date of issue of the certificate. Details of the medical authority issuing the certificate, including their signature and seal.Passport-sized attested photograph of the person with a disability (showing the face only). Proof of residence submitted by the applicant.'}, {'text': '###human: What is the significance of the Certificate of Disability? ###assistant: The Certificate of Disability holds significant importance as it officially recognizes the disability status of an individual. It enables them to access various government benefits, reservations, and facilities intended for people with disabilities.'}, {'text': '###human: Can individuals with other types of disabilities, apart from the ones mentioned, apply for the Certificate of Disability using this form? ###assistant: The form specifically covers individuals with locomotor disability, dwarfism, or blindness. Individuals with other types of disabilities may need to use different forms or procedures specific to their condition.'}, {'text': '###human: How is the percentage of permanent disability determined in the Certificate of Disability? ###assistant: The percentage of permanent disability is determined based on guidelines specified in the form. It is assessed in relation to a specific part of the body affected by the disability.'}, {'text': '###human: Can the Certificate of Disability be used for availing any special benefits or privileges? ###assistant: Yes, the Certificate of Disability can be used to avail various special benefits and privileges offered by the government for people with disabilities. These benefits may include reservations in education, employment, and welfare schemes.'}, {'text': '###human: Is the Certificate of Disability applicable nationwide, or does it vary by region? ###assistant: The Certificate of Disability generally applies nationwide. However, specific rules and procedures for obtaining the certificate may vary slightly between different regions or states.'}, {'text': '###human: How long is the Certificate of Disability valid once issued? ###assistant: The validity of the Certificate of Disability may vary based on local regulations. In many cases, the certificate is considered valid indefinitely unless specified otherwise or until the disability condition changes significantly.'}, {'text': '###human: What is the purpose of including a recent passport-sized photograph in the Certificate of Disability? ###assistant: Including a recent passport-sized photograph helps in identifying the person with the disability and ensures that the certificate is linked to the correct individual.'}]\n",
        "df = pd.DataFrame(json_data)"
      ],
      "metadata": {
        "id": "BfBgKwdRl_qr"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Step 1 data in form of questions and answers fetched from json file"
      ],
      "metadata": {
        "id": "uymskr5-bJ0s"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "ziH3DJ8RXWej"
      },
      "outputs": [],
      "source": [
        "dataset = datasets.Dataset.from_pandas(df)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rjOMoSbGSxx9"
      },
      "source": [
        "Installing required packages using GPU runtime type in colab for sharding"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eCv0Km_njEjv",
        "outputId": "2fa70ddd-f544-425e-c795-39e8381aceb0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: transformers==4.30.2 in /usr/local/lib/python3.10/dist-packages (4.30.2)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (3.12.2)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.14.1 in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (0.16.4)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (1.23.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (23.1)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (6.0.1)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (2023.6.3)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (2.31.0)\n",
            "Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (0.13.3)\n",
            "Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (0.3.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers==4.30.2) (4.66.1)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.14.1->transformers==4.30.2) (2023.6.0)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.14.1->transformers==4.30.2) (4.7.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.30.2) (3.2.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.30.2) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.30.2) (2.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.30.2) (2023.7.22)\n",
            "Requirement already satisfied: accelerate==0.21.0 in /usr/local/lib/python3.10/dist-packages (0.21.0)\n",
            "Requirement already satisfied: peft==0.4.0 in /usr/local/lib/python3.10/dist-packages (0.4.0)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from accelerate==0.21.0) (1.23.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from accelerate==0.21.0) (23.1)\n",
            "Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate==0.21.0) (5.9.5)\n",
            "Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from accelerate==0.21.0) (6.0.1)\n",
            "Requirement already satisfied: torch>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from accelerate==0.21.0) (2.0.1+cu118)\n",
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (from peft==0.4.0) (4.30.2)\n",
            "Requirement already satisfied: safetensors in /usr/local/lib/python3.10/dist-packages (from peft==0.4.0) (0.3.3)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate==0.21.0) (3.12.2)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate==0.21.0) (4.7.1)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate==0.21.0) (1.12)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate==0.21.0) (3.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate==0.21.0) (3.1.2)\n",
            "Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate==0.21.0) (2.0.0)\n",
            "Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.10.0->accelerate==0.21.0) (3.27.2)\n",
            "Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.10.0->accelerate==0.21.0) (16.0.6)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.14.1 in /usr/local/lib/python3.10/dist-packages (from transformers->peft==0.4.0) (0.16.4)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers->peft==0.4.0) (2023.6.3)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers->peft==0.4.0) (2.31.0)\n",
            "Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /usr/local/lib/python3.10/dist-packages (from transformers->peft==0.4.0) (0.13.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers->peft==0.4.0) (4.66.1)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.14.1->transformers->peft==0.4.0) (2023.6.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.10.0->accelerate==0.21.0) (2.1.3)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->peft==0.4.0) (3.2.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->peft==0.4.0) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->peft==0.4.0) (2.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->peft==0.4.0) (2023.7.22)\n",
            "Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.10.0->accelerate==0.21.0) (1.3.0)\n",
            "Requirement already satisfied: bitsandbytes==0.40.2 in /usr/local/lib/python3.10/dist-packages (0.40.2)\n",
            "Requirement already satisfied: trl==0.4.7 in /usr/local/lib/python3.10/dist-packages (0.4.7)\n",
            "Requirement already satisfied: torch>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from trl==0.4.7) (2.0.1+cu118)\n",
            "Requirement already satisfied: transformers>=4.18.0 in /usr/local/lib/python3.10/dist-packages (from trl==0.4.7) (4.30.2)\n",
            "Requirement already satisfied: numpy>=1.18.2 in /usr/local/lib/python3.10/dist-packages (from trl==0.4.7) (1.23.5)\n",
            "Requirement already satisfied: accelerate in /usr/local/lib/python3.10/dist-packages (from trl==0.4.7) (0.21.0)\n",
            "Requirement already satisfied: datasets in /usr/local/lib/python3.10/dist-packages (from trl==0.4.7) (2.14.4)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.4.0->trl==0.4.7) (3.12.2)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.4.0->trl==0.4.7) (4.7.1)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.4.0->trl==0.4.7) (1.12)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.4.0->trl==0.4.7) (3.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.4.0->trl==0.4.7) (3.1.2)\n",
            "Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.4.0->trl==0.4.7) (2.0.0)\n",
            "Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.4.0->trl==0.4.7) (3.27.2)\n",
            "Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.4.0->trl==0.4.7) (16.0.6)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.14.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.18.0->trl==0.4.7) (0.16.4)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.18.0->trl==0.4.7) (23.1)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.18.0->trl==0.4.7) (6.0.1)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.18.0->trl==0.4.7) (2023.6.3)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers>=4.18.0->trl==0.4.7) (2.31.0)\n",
            "Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.18.0->trl==0.4.7) (0.13.3)\n",
            "Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.18.0->trl==0.4.7) (0.3.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.18.0->trl==0.4.7) (4.66.1)\n",
            "Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate->trl==0.4.7) (5.9.5)\n",
            "Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets->trl==0.4.7) (9.0.0)\n",
            "Requirement already satisfied: dill<0.3.8,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets->trl==0.4.7) (0.3.7)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets->trl==0.4.7) (1.5.3)\n",
            "Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets->trl==0.4.7) (3.3.0)\n",
            "Requirement already satisfied: multiprocess in /usr/local/lib/python3.10/dist-packages (from datasets->trl==0.4.7) (0.70.15)\n",
            "Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.10/dist-packages (from datasets->trl==0.4.7) (2023.6.0)\n",
            "Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets->trl==0.4.7) (3.8.5)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl==0.4.7) (23.1.0)\n",
            "Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl==0.4.7) (3.2.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl==0.4.7) (6.0.4)\n",
            "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl==0.4.7) (4.0.3)\n",
            "Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl==0.4.7) (1.9.2)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl==0.4.7) (1.4.0)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl==0.4.7) (1.3.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers>=4.18.0->trl==0.4.7) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers>=4.18.0->trl==0.4.7) (2.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers>=4.18.0->trl==0.4.7) (2023.7.22)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.4.0->trl==0.4.7) (2.1.3)\n",
            "Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets->trl==0.4.7) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets->trl==0.4.7) (2023.3)\n",
            "Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.4.0->trl==0.4.7) (1.3.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas->datasets->trl==0.4.7) (1.16.0)\n"
          ]
        }
      ],
      "source": [
        "#cell1 run this\n",
        "!pip install transformers==4.30.2\n",
        "\n",
        "!pip install accelerate==0.21.0 peft==0.4.0\n",
        "!pip install bitsandbytes==0.40.2 trl==0.4.7"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Loading the llama2 model from huggingface"
      ],
      "metadata": {
        "id": "t1vzW1jwmfCj"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 49,
          "referenced_widgets": [
            "6898a15c5d3d4d418bf3137454ee900a",
            "3d18b37ba5054f5f9adabb3a5d6a95b2",
            "ac71e84ce41a4ebcade85eab93f28614",
            "20ff16fc1a7c48958a9bfad95830bdfa",
            "934e408a4fcd4c6487fac90fe63569ff",
            "fa6f60953dda4227b8db64d2fc3dad80",
            "d94b2935b3f34c23abb654d3f6da4c11",
            "3e440a903e524bff9361dc7000ecff42",
            "674881d070a04e91b3074e76a2cb991f",
            "797a378e764e43e1a061d2579f4f806d",
            "59a722f37ffb4d64b10b2b927f76fab4"
          ]
        },
        "id": "ZwXZbQ2dSwzI",
        "outputId": "7fb1ad6f-860e-4cf4-81ac-3c9e2f15d6e6"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "Loading checkpoint shards:   0%|          | 0/14 [00:00<?, ?it/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "6898a15c5d3d4d418bf3137454ee900a"
            }
          },
          "metadata": {}
        }
      ],
      "source": [
        "import torch\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer\n",
        "\n",
        "model_name = \"TinyPixel/Llama-2-7B-bf16-sharded\"\n",
        "#model_name=\"divya9103/llama2-divya\"\n",
        "bnb_config = BitsAndBytesConfig(\n",
        "    load_in_4bit=True,\n",
        "    bnb_4bit_quant_type=\"nf4\",\n",
        "    bnb_4bit_compute_dtype=torch.float16,\n",
        ")\n",
        "\n",
        "model = AutoModelForCausalLM.from_pretrained(\n",
        "    model_name,\n",
        "    quantization_config=bnb_config,\n",
        "    trust_remote_code=True\n",
        ")\n",
        "model.config.use_cache = False"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "XDS2yYmlUAD6"
      },
      "outputs": [],
      "source": [
        "tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, return_token_type_ids=False)\n",
        "tokenizer.pad_token = tokenizer.eos_token"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "model configuration parameters"
      ],
      "metadata": {
        "id": "sW14e8kUmZYK"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "xP4qZ0NPL1A-"
      },
      "outputs": [],
      "source": [
        "import logging\n",
        "logging.getLogger(\"transformers.modeling_utils\").setLevel(logging.ERROR)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dQdvjTYTT1vQ"
      },
      "outputs": [],
      "source": [
        "from peft import LoraConfig, get_peft_model\n",
        "\n",
        "lora_alpha = 16\n",
        "lora_dropout = 0.1\n",
        "lora_r = 64\n",
        "\n",
        "peft_config = LoraConfig(\n",
        "    lora_alpha=lora_alpha,\n",
        "    lora_dropout=lora_dropout,\n",
        "    r=lora_r,\n",
        "    bias=\"none\",\n",
        "    task_type=\"CAUSAL_LM\"\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dzsYHLwIZoLm"
      },
      "source": [
        "## Loading the trainer"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aTBJVE4PaJwK"
      },
      "source": [
        "Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OCFTvGW6aspE"
      },
      "outputs": [],
      "source": [
        "from transformers import TrainingArguments\n",
        "\n",
        "output_dir = \"./results\"\n",
        "per_device_train_batch_size = 2\n",
        "gradient_accumulation_steps = 4\n",
        "optim = \"paged_adamw_32bit\"\n",
        "save_steps = 100\n",
        "logging_steps = 10\n",
        "learning_rate = 2e-4\n",
        "max_grad_norm = 0.3\n",
        "max_steps = 100\n",
        "warmup_ratio = 0.03\n",
        "lr_scheduler_type = \"constant\"\n",
        "\n",
        "training_arguments = TrainingArguments(\n",
        "    output_dir=output_dir,\n",
        "    per_device_train_batch_size=per_device_train_batch_size,\n",
        "    gradient_accumulation_steps=gradient_accumulation_steps,\n",
        "    optim=optim,\n",
        "    save_steps=save_steps,\n",
        "    logging_steps=logging_steps,\n",
        "    learning_rate=learning_rate,\n",
        "    fp16=True,\n",
        "    max_grad_norm=max_grad_norm,\n",
        "    max_steps=max_steps,\n",
        "    warmup_ratio=warmup_ratio,\n",
        "    group_by_length=True,\n",
        "    lr_scheduler_type=lr_scheduler_type,\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "I3t6b2TkcJwy"
      },
      "source": [
        "Then finally pass everthing to the trainer"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TNeOBgZeTl2H"
      },
      "outputs": [],
      "source": [
        "from trl import SFTTrainer\n",
        "\n",
        "max_seq_length = 100\n",
        "\n",
        "trainer = SFTTrainer(\n",
        "    model=model,\n",
        "    train_dataset=dataset,\n",
        "    peft_config=peft_config,\n",
        "    dataset_text_field=\"text\",\n",
        "    max_seq_length=max_seq_length,\n",
        "    tokenizer=tokenizer,\n",
        "    args=training_arguments,\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GWplqqDjb3sS"
      },
      "source": [
        "We will also pre-process the model by upcasting the layer norms in float 32 for more stable training"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7OyIvEx7b1GT"
      },
      "outputs": [],
      "source": [
        "for name, module in trainer.model.named_modules():\n",
        "    if \"norm\" in name:\n",
        "        module = module.to(torch.float32)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1JApkSrCcL3O"
      },
      "source": [
        "## Train the model"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JjvisllacNZM"
      },
      "source": [
        "Now let's train the model! Simply call `trainer.train()`"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_kbS7nRxcMt7"
      },
      "outputs": [],
      "source": [
        "trainer.train()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lUBzVm5VqiCS"
      },
      "outputs": [],
      "source": [
        "trainer.save_model('/content/llama2')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "H5c0ppfasK29"
      },
      "source": [
        "During training, the model should converge nicely as follows:\n",
        "\n",
        "![image](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/loss-falcon-7b.png)\n",
        "\n",
        "The `SFTTrainer` also takes care of properly saving only the adapters during training instead of saving the entire model."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "v2CEgCF14M0m"
      },
      "outputs": [],
      "source": [
        "model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training\n",
        "model_to_save.save_pretrained(\"outputs\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qmA4G6C64dJ4"
      },
      "outputs": [],
      "source": [
        "lora_config = LoraConfig.from_pretrained('outputs')\n",
        "model = get_peft_model(model, lora_config)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Asking the finetuned model to generate output"
      ],
      "metadata": {
        "id": "J0wVe5nYnEKE"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pgt86z-x4diG"
      },
      "outputs": [],
      "source": [
        "#text = \"Écrire un texte dans un style baroque sur la glace et le feu ### Assistant: Si j'en luis éton\"\n",
        "\n",
        "text=\"What do candidates need to do for the Phase I Examination? ###assistant: Candidates need to\"\n",
        "device = \"cuda:0\"\n",
        "inputs = tokenizer(text, return_token_type_ids=False, return_tensors=\"pt\").to(device)\n",
        "outputs = model.generate(**inputs, max_new_tokens=10)\n",
        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Code for saving (few errors) but ignore from here"
      ],
      "metadata": {
        "id": "VybIv8Z5nKxa"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fjOInz-Q-54k"
      },
      "outputs": [],
      "source": [
        "# model.push_to_hub(\"llama2-divya\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "GbqWMREsOgjV"
      },
      "outputs": [],
      "source": [
        "\n",
        "from transformers import AutoModelForCausalLM\n",
        "\n",
        "# Replace \"username/model-name\" with the appropriate username and model name on the Hub\n",
        "model_name = \"Ak1139/ak\"\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "X2T8GUOn4QzR"
      },
      "outputs": [],
      "source": [
        "from transformers import LlamaForCausalLM, LlamaTokenizer\n",
        "from peft import PeftModel"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VZNGIGxM5wBg"
      },
      "outputs": [],
      "source": [
        "import locale\n",
        "locale.getpreferredencoding = lambda: \"UTF-8\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "SkQ_Ny3y5cTd"
      },
      "outputs": [],
      "source": [
        "!pip install SentencePiece\n",
        "!pip install timm"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a4ZaDN_YSzNS"
      },
      "outputs": [],
      "source": [
        "import timm\n",
        "model_reloaded = timm.create_model('hf_hub:nateraw/resnet18-random', pretrained=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "WNZyunnSkqYz"
      },
      "outputs": [],
      "source": [
        "trainer.save_model(\"path/to/model\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sMN2K44DWelr"
      },
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "\n",
        "# Mount Google Drive\n",
        "drive.mount('/content/drive')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rJ04DDRFJnvz"
      },
      "outputs": [],
      "source": [
        "torch.jit.save(torch.jit.script(model), \"model.pt\")\n",
        "\n",
        "\n",
        "# Load the model from TorchScript\n",
        "loaded_model = torch.jit.load(\"model.pt\")"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "provenance": [],
      "gpuType": "T4"
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
        "6898a15c5d3d4d418bf3137454ee900a": {
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
              "IPY_MODEL_3d18b37ba5054f5f9adabb3a5d6a95b2",
              "IPY_MODEL_ac71e84ce41a4ebcade85eab93f28614",
              "IPY_MODEL_20ff16fc1a7c48958a9bfad95830bdfa"
            ],
            "layout": "IPY_MODEL_934e408a4fcd4c6487fac90fe63569ff"
          }
        },
        "3d18b37ba5054f5f9adabb3a5d6a95b2": {
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
            "layout": "IPY_MODEL_fa6f60953dda4227b8db64d2fc3dad80",
            "placeholder": "​",
            "style": "IPY_MODEL_d94b2935b3f34c23abb654d3f6da4c11",
            "value": "Loading checkpoint shards:   0%"
          }
        },
        "ac71e84ce41a4ebcade85eab93f28614": {
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
            "bar_style": "",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_3e440a903e524bff9361dc7000ecff42",
            "max": 14,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_674881d070a04e91b3074e76a2cb991f",
            "value": 0
          }
        },
        "20ff16fc1a7c48958a9bfad95830bdfa": {
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
            "layout": "IPY_MODEL_797a378e764e43e1a061d2579f4f806d",
            "placeholder": "​",
            "style": "IPY_MODEL_59a722f37ffb4d64b10b2b927f76fab4",
            "value": " 0/14 [00:00&lt;?, ?it/s]"
          }
        },
        "934e408a4fcd4c6487fac90fe63569ff": {
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
        "fa6f60953dda4227b8db64d2fc3dad80": {
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
        "d94b2935b3f34c23abb654d3f6da4c11": {
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
        "3e440a903e524bff9361dc7000ecff42": {
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
        "674881d070a04e91b3074e76a2cb991f": {
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
        "797a378e764e43e1a061d2579f4f806d": {
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
        "59a722f37ffb4d64b10b2b927f76fab4": {
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