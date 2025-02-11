# Rice Classification ML

## Description  
This project uses machine learning to classify rice species (**Jasmine vs. Gonen**) based on geometric features.  
The model is built using **Random Forest Classifier** and evaluates its performance using accuracy and a confusion matrix.

## Dataset  
The dataset contains the following features:  

- `Area`  
- `MajorAxisLength`  
- `MinorAxisLength`  
- `Eccentricity`  
- `ConvexArea`  
- `EquivDiameter`  
- `Extent`  
- `Perimeter`  
- `Roundness`  
- `AspectRation`  
- `Class` (label: `1` - Jasmine, `0` - Gonen)  

## Requirements  
To run the code, install the required Python libraries:  

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
