# echo "Launcher for NarrativePipelineQA"
# echo "Author        : Hoang Le"
# echo "Institution   : VinAI"

# echo "0.1. Decompose each document in dataset into paragraphs."
# python -m modules.data_reading.data_reading

# echo "0.2. Find golden passages for each question and trigger training of ParasSelection"
python -m modules.paras_selection.paras_selection --num_proc 5