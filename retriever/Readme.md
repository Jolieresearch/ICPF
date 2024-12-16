### Dataset Download

**[MicroLens]** Please refer to this repo: https://github.com/westlake-repl/MicroLens, select **MicroLens-100k-Dataset** and download the **raw videos** with their **titles (en)**.

### Data Preprocess

#### Create data.pkl

Create a dataframe in pickle format, named data.pkl, with the following columns:

- item_id : This columns is the id of each micro-video, which can be the name of each video without '.mp4' suffix.
- text: This columns is the title of each micro-video.
- label: The label popularity here is defined as the number of total comments for a micro-video.

##### Feature extraction

Run the **video_frame_capture.py** for each raw micro-videos obtain video frames, here the default number of frames is 10 for each video.

Then run the **extract_features.py** to extract the video frames and text features.

### Retrieval Preprocess

Run the **retriever.py** to do retrieval process.

Then run the **stack.py** to stack the relevant video frames and text features.
