# Covid - 19 Dashboard


This Project is built on Flask 

## Creating a Virtual Environment 
```sh
$pip install virtualenv
$virtualenv Covid-19_Dashboard
```

Windows
```sh
$<path>Covid-19_Dashboard\Scripts\activate
```

Mac OS / Linux
```sh
$source <path>Covid-19_Dashboard/bin/activate
```

Install the Required Packages from Requirements.txt
```sh
$pip3 install -r requirements.txt
```

## To get the latest Pridiction values

```sh
$python Merged_ARIMA.py

$python Merged_Vacc.py

$python Merged_Prophet.py
```

## To run Dashboard
```sh
$python covid19.py
```
