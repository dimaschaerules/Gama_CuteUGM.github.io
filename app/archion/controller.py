from app import app
from flask import render_template, request, redirect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from os.path import exists
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import os
import sqlite3
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for, session,flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re



class RandomForest:
    
    # constructor (akan dipanggil saat class di instansiasi / dibuatnya objek)
    def __init__(self, data = None): # memberikan nilai default kepada variabel data agar tidak perlu menbambah data baru
        self.__location = 'app/pretrained/data_pretrained.xlsx' # lokasi menyimpan file database
        self.document = None # variabel penampung untuk dokumen
        data_exist = False
        try:
            self.data = data.copy() # menyalin isi variabel (parameter) data ke dalam variabel (penampung) data
            data_exist = True
        except:
            self.data = None
            
        self._X = None # Variabel penampung _X (features)
        self._y = None # Variabel penampung -y (labels)
        self._sc = StandardScaler() # Variabel penampung untuk normalisasi (standard scaler)
        
        if exists(self.__location): # jika ada file pretrained / database sebelumnya
            pretrained_dataset = pd.read_excel(self.__location) # load file pretrained atau database ke dalam variabel pretrained_dataset
            
            X_pretrained = pretrained_dataset.drop(['Sample', 'Diagnosis'], axis=1) # ekstrak features dari pretrained_dataset
            y_pretrained = pretrained_dataset['Diagnosis'] # ekstrak labels dari pretrained_dataset
            
            try:
                if data != None: # jika parameter data ada
                    X = data.drop(['Sample', 'Diagnosis'], axis=1) # ekstrak features dari data
                    y = data['Diagnosis'] # ekstrak labels dari data

                    X_pretrained = X_pretrained.append(X, ignore_index=True) # gabungkan features dari data ke dalam kumpulan features pretrained_dataset
                    y_pretrained = y_pretrained.append(y, ignore_index=True) # gabungkan labels dari data ke dalam kumpulan labels pretrained_dataset
            except:
                if not data.empty:
                    X = data.drop(['Sample', 'Diagnosis'], axis=1) # ekstrak features dari data
                    y = data['Diagnosis'] # ekstrak labels dari data

                    X_pretrained = X_pretrained.append(X, ignore_index=True) # gabungkan features dari data ke dalam kumpulan features pretrained_dataset
                    y_pretrained = y_pretrained.append(y, ignore_index=True) # gabungkan labels dari data ke dalam kumpulan labels pretrained_dataset
                    
            # melakukan normalisasi dengan standard scaler
            X_train = self._sc.fit_transform(X_pretrained)
            
            # menyiapkan algoritma untuk melakukan random forest dengan jumlah estimator 500, kriteria nya sebagai entropy dan random state = 0
            self._classifier = RandomForestClassifier(n_estimators = 400, criterion = 'entropy', random_state = 0)
            self._classifier.fit(X_train, y_pretrained) # melakukan training dari pretrained_dataset diatas
            
            current_accuracy = 0
            try:
                current_accuracy = self.accuracy
            except:
                current_accuracy = 0
            
            try:
                if data != None: # jika paramater data ada
                    y_pred = self._classifier.predict(X) # melakukan prediksi y_pred
                    self.accuracy = metrics.accuracy_score(y, y_pred) # menghitung akurasi dan menyimpan ke dalam variabel accuracy
                else:
                    y_pred = self._classifier.predict(X_train)
                    self.accuracy = cross_val_score(self._classifier, X_train, y_pred, cv=2).mean()                    
            except:
                if not data.empty:
                    y_pred = self._classifier.predict(X) # melakukan prediksi y_pred
                    self.accuracy = metrics.accuracy_score(y, y_pred) # menghitung akurasi dan menyimpan ke dalam variabel accuracy
                else:
                    y_pred = self._classifier.predict(X_train)
                    self.accuracy = cross_val_score(self._classifier, X_train, y_pred, cv=2).mean()
            
            self._X = X_pretrained.copy() # menyalin isi dari variabel X_pretrained ke dalam variabel penampung X
            self._y = y_pretrained.copy() # menyalin isi dari variabel y_pretrained ke dalam vairabel penampung y
            
            new_data = pretrained_dataset.append(data, ignore_index=True) # menggabungkan dokumen baru ke dalam pretrained dataset / database
            
            if current_accuracy < self.accuracy:
                new_data.to_excel(self.__location, index=False) # export dokumen baru tersebut ke pretrained dataset
            
        elif data_exist: # jika variabel data ada namun pretrained dataset belum ada
            X = data.drop(['Sample', 'Diagnosis'], axis=1) # ekstrak features dari variabel data
            y = data['Diagnosis'] # ekstrak labels dari variabel data

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0) # split data untuk training dan testing

            X_train = self._sc.fit_transform(X_train) # melakukan normalisasi untuk data training nya
            X_test = self._sc.transform(X_test) # melakukan normalisasi untuk data testing nya

            # menyiapkan algoritma untuk melakukan random forest dengan jumlah estimator 400, kriteria nya sebagai entropy dan random state = 0
            self._classifier = RandomForestClassifier(n_estimators = 400, criterion = 'entropy', random_state = 0)
            self._classifier.fit(X_train, y_train) # melakukan training dari pretrained_dataset diatas
            y_pred = self._classifier.predict(X_test) # melakukan prediksi data test untuk mendapatkan akurasi
            
            self.accuracy = metrics.accuracy_score(y_test, y_pred) # menghitung akurasi dari hasil prediksi dan label data test
            
            self._X = X.copy() # menyalin isi dari variabel X ke dalam variabel penampung X
            self._y = y.copy() # menyalin isi dari variabel y ke dalam variabel penampung y

            data.to_excel(self.__location, index=False) # ekspor dokumen tersebut ke pretrained dataset
            
        else: # jika variabel data tidak ada dan dokumen pretrained sebelumnya tidak ada
            print("Belum ada data yang di-train")
            return [] # kembalikan list kosong

        print("Data sudah dilatih...") # pesan akan ditampilkan jika training berhasil dijalankan
        
    # membuat method untuk melakukan predict 1 dokumen excel
    def predict_document(self, data):
        new_document = data.drop(['Sample', 'Diagnosis'], axis=1) # ekstrak features dari variabel data

        new_document = self._sc.transform(new_document) # melakukan normalisasi untuk data features nya

        results = self._classifier.predict(new_document) # melakukan prediksi dari data features di dokumennya

        data['Diagnosis'] = results # menambahkan kolom diagnosa dari hasil prediksi dokumen tersebut
        
        self.document = data.copy() # menyalin data dokumen yang telah diprediksi ke dalam variabel penampung document
        
        accuracy_test = cross_val_score(self._classifier, new_document, results, cv=2).mean()
        
        if exists(self.__location and accuracy_test > self.accuracy): # jika pretrained dataset belum ada 
            pretrained_dataset = pd.read_excel(self.__location) # load dataset pretrained data ke dalam variabel pretrained_dataset
            
            new_dataset = pretrained_dataset.append(data, ignore_index=True) # menggabungkan ke dokumen yang telah diprediksi ke dalam variabel dataset pretrained
            
            new_dataset.to_excel(self.__location, index=False) # export dokumen tersebut ke dalam pretrained dataset (database)
        
        return self.document # mengembalikan dokumen
        
    # method untuk melakukan prediksi single data
    def predict(self, data):
        val_data = np.array(data) # transformasi list menjadi array agar bisa di reshape
        
        if len(val_data.shape) == 1: # jika shape masih 1 (vektor)
            val_data = np.reshape(val_data, [1, -1]) # ubah bentuknya menjadi matriks
            
        val_data = self._sc.transform(val_data) # normalisasi data tersebut
        
        results = self._classifier.predict(val_data) # melakukan prediksi dari single data
        
        return results # mengambalikan hasil prediksi nya
        
    # melakukan ekspor dokumen yang telah diprediksi menjadi file terpisah
    def data_export(self, namafile):
        try:
            self.document.to_excel(namafile, index=False) # ekspor dokumen excel
            
            print(f"Berhasil export data ke excel '{namafile}'") # menampilkan pesan jika dokumen berhasil ter-ekspor
        except:
            print("Gagal melakukan export file") # menampilkan pesan jika gagal di-ekspor


@app.route("/dashboard", methods=["GET"])
def index():
    dataset_asli = pd.read_excel(r'/Users/heyiamr/Database_Gamacute/dataset_aslinew.xlsx')
    dataset_latih = pd.read_excel(r'/Users/heyiamr/Document_DataLatih/DataLatihNew.xlsx')
    index = dataset_latih.index
    number_of_traindata = len(index)
    print(number_of_traindata)
    index2 = dataset_asli.index
    number_of_database = len(index2)
    print(number_of_database)
    labelencoder = LabelEncoder()
    dataset_asli['Diagnosis'] = labelencoder.fit_transform(dataset_asli['Diagnosis'])
    database = dataset_asli.drop(["Sample"], axis=1)
    x = database.drop(["Diagnosis"], axis=1)
    y = database["Diagnosis"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    classifier = RandomForestClassifier(n_estimators = 400, criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    dataset_tambahan = pd.read_excel(r'/Users/heyiamr/Document_TambahDataTraining/DataLatihAsli.xlsx')
    index = dataset_latih.index
    number_of_rows = len(index)
    number_of_rows
    return render_template("index.html", numberdata = number_of_rows)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", modal="hide")
    else:
        datauji_dict = {
            'Sample': request.form["sample"],
            'RBC' : float(request.form["rbc"]),
            'HGB' : float(request.form["hgb"]),
            'HCT': float(request.form["hct"]), 
            'MCV': float(request.form["mcv"]), 
            'MCH': float(request.form["mch"]), 
            'MCHC': float(request.form["mchc"]), 
            'RDW': float(request.form["rdw"]),
            }
        datauji_dict

        test_dict = {
            'RBC' : float(request.form["rbc"]),
            'HGB' : float(request.form["hgb"]),
            'HCT': float(request.form["hct"]), 
            'MCV': float(request.form["mcv"]), 
            'MCH': float(request.form["mch"]), 
            'MCHC': float(request.form["mchc"]), 
            'RDW': float(request.form["rdw"]),
            }
        new_test = test_dict
        data1 = list(new_test.values())
        data2 = list(datauji_dict.values())
        an_array = np.array(data1)
        rf = RandomForest()
        new_data = rf.predict(an_array)
        new_data = new_data.astype(int)
        new_data1 = new_data
        condlist = [new_data1 == 0, new_data1 == 1, new_data1 == 2, new_data1 == 3]
        choicelist = ['BTT', 'HbE', 'IDA', 'MIX (BTT and DB or HbE and DB)']
        new_data = np.select(condlist, choicelist, default='unknown')
        df = pd.DataFrame(new_data, columns=['Diagnosis'])
        an_array = np.array(data2)
        new_array = np.append(an_array, df)
        an_array2 = pd.DataFrame(new_array)
        new_array2 = pd.DataFrame(new_array)
        new_array2_transposed = new_array2.T
        df_new = new_array2_transposed.rename(columns={0: 'Sample',
                                                       1: 'RBC',
                                                       2: 'HGB',
                                                       3: 'HCT',
                                                       4: 'MCV',
                                                       5: 'MCH',
                                                       6: 'MCHC',
                                                       7: 'RDW',
                                                       8: 'Diagnosis'})

        dataset_ujisingle = df_new.to_excel(r'/Users/heyiamr/Document_DataUjiSingle/HasilDataUjiSingleNew.xlsx', index=False, engine='xlsxwriter')
        dataset_ujisingle = pd.read_excel(r'/Users/heyiamr/Document_DataUjiSingle/HasilDataUjiSingleNew.xlsx')
        database = pd.read_excel(r'/Users/heyiamr/Document_DataUjiSingle/DatabaseDataUjiSIngle.xlsx')
        new_df = pd.read_excel(r'/Users/heyiamr/Document_DataUjiSingle/DatabaseDataUjiSIngle.xlsx')

        for i, row in dataset_ujisingle.iterrows():
            df = pd.DataFrame(dataset_ujisingle)
            new_df = pd.concat([new_df, df], sort=False)
            new_df.to_excel(r'/Users/heyiamr/Document_DataUjiSingle/DatabaseDataUjiSingle.xlsx', index=False, engine='xlsxwriter')
        new_df
            
        dataset_ujisingle = df_new.to_excel(r'/Users/heyiamr/Document_DataUjiSingle/DataUjiSingleNew.xlsx', index=False)
        return render_template("predict.html", modal="show", Diagnosis=df["Diagnosis"].tolist())


@app.route("/bulk_predict", methods=["GET", "POST"])
def bulk_predict():
    if request.method == "GET":
        return render_template("bulk_predict.html")
    else:
        file = request.files["file"]
        file.save(os.path.join('app/db', 'dataset_raw.xlsx')) #save datauji

        dataset_asli = pd.read_excel(r'/Users/heyiamr/Document_DataLatih/Database_GAMACUTE.xlsx')
        dataset = pd.read_excel(r'/Users/heyiamr/Document_DataLatih/DataLatih4Kelas copy.xlsx')
        labelencoder = LabelEncoder()
        dataset['Diagnosis'] = labelencoder.fit_transform(dataset['Diagnosis'])
        rf = RandomForest(dataset)
        document = pd.read_excel(r'/Users/heyiamr/Document_DataUji/DataUji.xlsx')
        document_result = rf.predict_document(document)
        rf.data_export(r'/Users/heyiamr/Document_DatabaseHasilPrediksi/predicted_document_new.xlsx')
        dataset_uji = pd.read_excel(r'/Users/heyiamr/Document_DatabaseHasilPrediksi/predicted_document_new.xlsx')
        
        for i, row in dataset_uji.iterrows():
            df = pd.DataFrame(dataset_uji)
            new_df = pd.concat([dataset_asli, df], sort=False)
            new_df.to_excel(r'/Users/heyiamr/Database_Gamacute/dataset_asli.xlsx', index=False, engine='xlsxwriter')

        dataset_uji.replace(0,
           "Beta Thalassemia Trait", 
           inplace=True)
        dataset_uji.replace(1,
           "Hemoglobin E", 
           inplace=True)
        dataset_uji.replace(2,
           "Iron Deficiency Anemia", 
           inplace=True)
        dataset_uji.replace(3,
           "MIX (BTT and DB or HbE and DB)", 
           inplace=True)
        dataset_uji
        
        new_df.replace(0,
           "Beta Thalassemia Trait", 
           inplace=True)
        new_df.replace(1,
           "Hemoglobin E", 
           inplace=True)
        new_df.replace(2,
           "Iron Deficiency Anemia", 
           inplace=True)
        new_df.replace(3,
           "MIX (BTT and DB or HbE and DB)", 
           inplace=True)
        new_df

        df = pd.DataFrame(new_df)
        df.to_excel(r'/Users/heyiamr/Document_DatabaseHasilPrediksi/Database_GAMACUTE.xlsx', index = False)
        df2 = pd.DataFrame(dataset_uji)
        df2.to_excel(r'/Users/heyiamr/Document_DatabaseHasilPrediksi/predicted_document_new.xlsx', index = False)
        return render_template("result_bulk_predict.html", tables=[df2.to_html(classes="table table-bordered align-items-center", index=False)])


@app.route("/training", methods=["GET", "POST"])
def training():
    if request.method == "GET":
        return render_template("input_data_training.html")
    else:
        file = request.files["file"]

        dataset_latih = pd.read_excel(r'/Users/heyiamr/Document_TambahDataTraining/Database_Utama.xlsx')
        dataset_latih_tambahan = pd.read_excel(r'/Users/heyiamr/Document_DataTrainingTambahan/DataLatihTambahan1.xlsx')

        for i, row in dataset_latih_tambahan.iterrows():
            df = pd.DataFrame(dataset_latih_tambahan)
            new2_df = pd.concat([dataset_latih, df], sort=False)
            new2_df.to_excel(r'/Users/heyiamr/Document_TambahDataTraining/DataLatihAsli.xlsx', index=False, engine='xlsxwriter')
            new2_df

        df = pd.DataFrame(new2_df)
        df.to_excel(r'/Users/heyiamr/Document_TambahDataTraining/DataLatihNew.xlsx', index = False)
        return render_template("result_tambah_data_latih.html", tables=[df.to_html(classes="table table-bordered align-items-center", index=False)])

@app.route("/information", methods=["GET", "POST"])
def information():
    if request.method == "GET":
        return render_template("information.html")
    else:
        return render_template("information.html")

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'gamacute'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '29061998Rd$'
app.config['MYSQL_DB'] = 'gamacutelogin'

# Intialize MySQL
mysql = MySQL(app)

# http://localhost:5000/ - this will be the login page, we need to use both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            flash("Incorrect username/password!", "danger")
    return render_template('login.html',title="Login")

# This will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM accounts WHERE username = %s', (username))
        cursor.execute( "SELECT * FROM accounts WHERE username LIKE %s", [username] )
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Incorrect username/password!", "danger")
        else:
        # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            flash("You have successfully registered!", "success")
            return redirect(url_for('login'))

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash("Please fill out the form!", "danger")
    # Show registration form with message (if any)
    return render_template('register.html',title="Register")


# This will be the home page, only accessible for loggedin users
@app.route('/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('index.html', username=session['username'],title="Home")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))    


@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

if __name__ =='__main__':
	app.run(Debug=True)
