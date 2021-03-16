#!/bin/bash

# echo "Launcher for NarrativePipelineQA"
# echo "Author        : Hoang Le"
# echo "Institution   : VinAI"

launch()
{
    # echo "1. Decompose each document in dataset into paragraphs."
    python -m modules.data_reading.data_reading --num_proc 5

    # echo "2. Start training"
    python -m modules.narrativepieline.NarrativePipeline  --num_proc 5
}


cloneProject()
{
    if [ -z $1 ]
    then
        sudo git clone https://github.com/tommyjohn1001/NarrativeGraphPipeline.git
    else
        sudo git clone --branch $1 https://github.com/tommyjohn1001/NarrativeGraphPipeline.git
    fi
}

updateSourceFromGit()
{
    branch_name="master"
    if [ -z $1 ]
    then
        branch_name="master"
    else
        branch_name=$1
    fi

    sudo git fetch --all
    sudo git reset --hard origin/$branch_name
}

debug()
{
    python -m $1 --num_proc 5
    jupyter notebook
}

print_help()
{
    echo "** Usage of Laucher"
    echo "laucher [] []"
    echo "-l or --launch or no flag : Launch the api"
    echo "-u or --update-git        : Update git"
    echo "                            if second argument is not specified, update is done with 'master' branch"
    echo "-c or --clone             : Clone git"
    echo "                            if second argument is not specified, clone is done with 'master' branch"
    echo "-h or --help              : Show help"
}

case $1 in
    -c | --clone )
        if [ -z $2 ] 
        then
            cloneProject
        else
            cloneProject $2
        fi        
        ;;
    -d | --debug )
        debug $2
        ;;
    -u | --update-git )
        if [ -z $2 ] 
        then
            updateSourceFromGit
        else
            updateSourceFromGit $2
        fi        
        ;;
    -h | --help )
        print_help
        ;;
    * )
        launch
        ;;
esac