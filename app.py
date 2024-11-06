import os

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename

# Инициализация приложения Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Загрузка обученной модели
model = load_model('model/fashion.h5')


# Функция для проверки допустимых расширений файлов
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Главная страница
@app.route("/")
def index():
    return render_template('index.html')


# Обработка загрузки файла и предсказания
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Обработка изображения для модели
        img = keras_image.load_img(filepath, target_size=(28, 28), color_mode="grayscale")
        img_array = keras_image.img_to_array(img)

        # Изменение формы на (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)

        img_array = 255 - img_array  # Инвертирование цветов
        img_array /= 255  # Нормализация

        # Предсказание
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Названия классов Fashion MNIST
        class_names = ['Футболка', 'Брюки ', 'Cвитер', 'Платье', 'Пальто ',
                       'Сандали', 'Рубашка ', 'Кроcсовки', 'Сумка', 'Ботильоны ']

        # Результат
        predicted_label = class_names[predicted_class]

        # Отображение результата
        return render_template('result.html', filename=filename, predicted_label=predicted_label)

    return redirect(request.url)


# Функция для отображения загруженного изображения
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))


# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True)