# Covid-19_Classification_Streamlit
CXR Classification with **streamlit**

## Datasets:
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
* https://github.com/abzargar/COVID-Classifier/archive/refs/heads/master.zip
* https://www.kaggle.com/sid321axn/covid-cxr-image-dataset-research

## Models:

 ![MobileNetV2](covid19_classification/Images/MobileNetV2.png)
 ![EfficientNetB0](covid19_classification/Images/EfficientNetB0.png)

## Usage

#### Clone the repository
```bash
git clone https://gitlab.aimedic.co/parsa592323/cxr.git
cd covid19_classification/src
```

#### Installing dependencies
```bash
chmod +x dependencies.sh
./dependencies.sh
```

#### Run the Service
```bash
chmod +x run.sh
./run.sh
```
#### Dockerfile

Create image

```bash
docker build -t 'docker-image' .
```
See created image

```bash
docker images
```

run image

```bash
docker run 'dockerfile-imagename'
```


#### Refer to http://localhost:8501

#### Results:

Upload An Image:
![1](covid19_classification/Images/upload_image.png)

The sample:
![2](covid19_classification/Images/COVID-19.jpeg)

Get the Classification Result:
![3](covid19_classification/Images/result.png)



