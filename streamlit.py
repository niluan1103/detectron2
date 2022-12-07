import streamlit as st
from PIL import Image
import wrist_detect
import numpy as np

if __name__ == '__main__':
    st.header('LTLab')
    st.title('Wrist Fracture Detection')
    st.subheader("using Faster-RCNN")
    img_file = st.file_uploader('Upload X-ray Image',type=['jpg','png','jpeg'])
    if img_file is not None:
        img = Image.open(img_file)
        caption = 'Image file uploaded: ' + img_file.name.split(".")[0]
        st.image(img, caption)

        st.text('Inferencing')
        img = np.array(img)
        visualized_img = wrist_detect.for_streamlit(img)
        st.image(visualized_img,'Visualized prediction')
