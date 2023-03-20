import streamlit as st
from PIL import Image
from keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
import requests
from bs4 import BeautifulSoup

# ***************************************Machine Learning Project**********************************************
# **********************************Fruit Recognition using Fruit360 Dataset************************************


model = tf.keras.models.load_model('model_saved.h5')

target_names = ['Apple Braeburn',
                'Apple Crimson Snow',
                'Apple Golden 1',
                  'Apple Golden 2',
                'Apple Golden 3',
                'Apple Granny Smith',
                'Apple Pink Lady',
                'Apple Red 1',
                'Apple Red 2',
                'Apple Red 3',
                'Apple Red Delicious',
                'Apple Red Yellow 1',
                'Apple Red Yellow 2',
                'Apricot',
                'Avocado',
                'Avocado ripe',
                'Banana',
                'Banana Lady Finger',
                'Banana Red',
                'Beetroot',
                'Blueberry',
                'Cactus fruit',
                'Cantaloupe 1',
                'Cantaloupe 2',
                'Carambula',
                'Cauliflower',
                'Cherry 1',
                'Cherry 2',
                'Cherry Rainier',
                'Cherry Wax Black',
                'Cherry Wax Red',
                'Cherry Wax Yellow',
                'Chestnut',
                'Clementine',
                'Cocos',
                'Corn',
                'Corn Husk',
                'Cucumber Ripe',
                'Cucumber Ripe 2',
                'Dates',
                'Eggplant',
                'Fig',
                'Ginger Root',
                'Granadilla',
                'Grape Blue',
                'Grape Pink',
                'Grape White',
                'Grape White 2',
                'Grape White 3',
                'Grape White 4',
                'Grapefruit Pink',
                'Grapefruit White',
                'Guava',
                'Hazelnut',
                'Huckleberry',
                'Kaki',
                'Kiwi',
                'Kohlrabi',
                'Kumquats',
                'Lemon',
                'Lemon Meyer',
                'Limes',
                'Lychee',
                'Mandarine',
                'Mango',
                'Mango Red',
                'Mangostan',
                'Maracuja',
                'Melon Piel de Sapo',
                'Mulberry',
                'Nectarine',
                'Nectarine Flat',
                'Nut Forest',
                'Nut Pecan',
                'Onion Red',
                'Onion Red Peeled',
                'Onion White',
                'Orange',
                'Papaya',
                'Passion Fruit',
                'Peach',
                'Peach 2',
                'Peach Flat',
                'Pear',
                'Pear 2',
                'Pear Abate',
                'Pear Forelle',
                'Pear Kaiser',
                'Pear Monster',
                'Pear Red',
                'Pear Stone',
                'Pear Williams',
                'Pepino',
                'Pepper Green',
                'Pepper Orange',
                'Pepper Red',
                'Pepper Yellow',
                'Physalis',
                'Physalis with Husk',
                'Pineapple',
                'Pineapple Mini',
                'Pitahaya Red',
                'Plum',
                'Plum 2',
                'Plum 3',
                'Pomegranate',
                'Pomelo Sweetie',
                'Potato Red',
                'Potato Red Washed',
                'Potato Sweet',
                'Potato White',
                'Quince',
                'Rambutan',
                'Raspberry',
                'Redcurrant',
                'Salak',
                'Strawberry',
                'Strawberry Wedge',
                'Tamarillo',
                'Tangelo',
                'Tomato 1',
                'Tomato 2',
                'Tomato 3',
                'Tomato 4',
                'Tomato Cherry Red',
                'Tomato Heart',
                'Tomato Maroon',
                'Tomato Yellow',
                'Tomato not Ripened',
                'Walnut',
                'Watermelon']


def fetch_calories(prediction):
    url = 'https://www.google.com/search?&q=calories in ' + prediction
    req = requests.get(url).text
    scrap = BeautifulSoup(req, 'html.parser')
    calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    return calories


def prepare_image(filepath):
    image = Image.open(filepath)
    image = image.resize((100, 100))
    image_array = np.array(image) / 255.0
    reshaped_image = image_array.reshape((1, 100, 100, 3))
    return reshaped_image


def run():
    st.title("Fruit Recognition")

    img_file = st.file_uploader("Choose am Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((100, 100))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            image = prepare_image("./upload_images/" + img_file.name)

            result = model.predict(image)[0]
            pnb = np.argmax(result)
            print(target_names[pnb])
            st.success(target_names[pnb])
            cal = fetch_calories(target_names[pnb])
            st.warning(cal)

run()
