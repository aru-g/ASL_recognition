# Detection and Identification of American Sign Language Letters in Real time videos

This project is broadly divided into two parts - 
1. Using YOLO model to detect the hands in the video frame
2. Using our CNN model to identify which ASL letter is being shown using the hand gesture

For training our CNN, we used this dataset from Kaggle -> 

For the first part of the project, we used a pre-trained YOLO model for hand detection from this github repo -> https://github.com/cansik/yolo-hand-detection

I built the UI using Streamlit, and will plan on deploying it to Streamlit-Share

For running this project on your local, simply clone the project.

1. Install all required packages using - 
```
$pip install -r requirements.txt
```
4. Download the model weights using -
```
# mac / linux
cd models && sh ./download-models.sh

# windows
cd models && powershell .\download-models.ps1
```
3. Run the streamlit app using - 
```
$streamlit run app.py
```

The app should open up on your local browser, else you can browse to http://localhost:8503 on your browser to see it.