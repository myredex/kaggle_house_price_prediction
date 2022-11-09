# House Prices prediction

In this notebook I'll try to predict price of the houses using machine learning techniques


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv
    /kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
    /kaggle/input/house-prices-advanced-regression-techniques/train.csv
    /kaggle/input/house-prices-advanced-regression-techniques/test.csv
    

<a id="description"></a>
# Data description

MSSubClass: Identifies the type of dwelling involved in the sale.	

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

MSZoning: Identifies the general zoning classification of the sale.
		
       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density
	
LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved
       	
Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
		
LotShape: General shape of property

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular
       
LandContour: Flatness of the property

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression
		
Utilities: Type of utilities available
		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	
	
LotConfig: Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property
	
LandSlope: Slope of property
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
	
Neighborhood: Physical locations within Ames city limits

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
			
Condition1: Proximity to various conditions
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
Condition2: Proximity to various conditions (if more than one is present)
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
BldgType: Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
	
HouseStyle: Style of dwelling
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level
	
OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
		
YearBuilt: Original construction date

YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

RoofStyle: Type of roof

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed
		
RoofMatl: Roof material

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
		
Exterior1st: Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
Exterior2nd: Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
	
MasVnrArea: Masonry veneer area in square feet

ExterQual: Evaluates the quality of the material on the exterior 
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
ExterCond: Evaluates the present condition of the material on the exterior
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
Foundation: Type of foundation
		
       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete	
       Slab	Slab
       Stone	Stone
       Wood	Wood
		
BsmtQual: Evaluates the height of the basement

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
		
BsmtCond: Evaluates the general condition of the basement

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
	
BsmtExposure: Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
	
BsmtFinType1: Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
		
BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Rating of basement finished area (if multiple types)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating
		
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace	
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace
		
HeatingQC: Heating quality and condition

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
CentralAir: Central air conditioning

       N	No
       Y	Yes
		
Electrical: Electrical system

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix	Mixed
		
1stFlrSF: First Floor square feet
 
2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

Kitchen: Kitchens above grade

KitchenQual: Kitchen quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       	
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality (Assume typical unless deductions are warranted)

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only
		
Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
		
GarageType: Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
		
GarageYrBlt: Year garage was built
		
GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
		
GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
GarageCond: Garage condition

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
PavedDrive: Paved driveway

       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
		
WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
		
Fence: Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
	
MiscFeature: Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
		
MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
		
SaleCondition: Condition of sale

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)

# Import libraries 


```python
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
```

# Datasets preparation


```python
# Load train dataframe
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
print(train.shape)
```

    (1460, 81)
    


```python
# Load train dataframe
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



As I want to prepare all the data to future exploration and analysis I'll join two dataframes and split them after preparation. So, I need to remember that test dataframe has ids 1461 and above and has no 'SalePrice' column.


```python
# Add 'SalePrice' column with 0 values
test['SalePrice'] = 0

# Check for df's shape
test.shape
```




    (1459, 81)




```python
# Append train df with test df
df = train.append(test)

# Check for df's shape
df.shape
```




    (2919, 81)



Now we have common dataframe to work with. Let's continue

Explore dataframe


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2919 entries, 0 to 1458
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             2919 non-null   int64  
     1   MSSubClass     2919 non-null   int64  
     2   MSZoning       2915 non-null   object 
     3   LotFrontage    2433 non-null   float64
     4   LotArea        2919 non-null   int64  
     5   Street         2919 non-null   object 
     6   Alley          198 non-null    object 
     7   LotShape       2919 non-null   object 
     8   LandContour    2919 non-null   object 
     9   Utilities      2917 non-null   object 
     10  LotConfig      2919 non-null   object 
     11  LandSlope      2919 non-null   object 
     12  Neighborhood   2919 non-null   object 
     13  Condition1     2919 non-null   object 
     14  Condition2     2919 non-null   object 
     15  BldgType       2919 non-null   object 
     16  HouseStyle     2919 non-null   object 
     17  OverallQual    2919 non-null   int64  
     18  OverallCond    2919 non-null   int64  
     19  YearBuilt      2919 non-null   int64  
     20  YearRemodAdd   2919 non-null   int64  
     21  RoofStyle      2919 non-null   object 
     22  RoofMatl       2919 non-null   object 
     23  Exterior1st    2918 non-null   object 
     24  Exterior2nd    2918 non-null   object 
     25  MasVnrType     2895 non-null   object 
     26  MasVnrArea     2896 non-null   float64
     27  ExterQual      2919 non-null   object 
     28  ExterCond      2919 non-null   object 
     29  Foundation     2919 non-null   object 
     30  BsmtQual       2838 non-null   object 
     31  BsmtCond       2837 non-null   object 
     32  BsmtExposure   2837 non-null   object 
     33  BsmtFinType1   2840 non-null   object 
     34  BsmtFinSF1     2918 non-null   float64
     35  BsmtFinType2   2839 non-null   object 
     36  BsmtFinSF2     2918 non-null   float64
     37  BsmtUnfSF      2918 non-null   float64
     38  TotalBsmtSF    2918 non-null   float64
     39  Heating        2919 non-null   object 
     40  HeatingQC      2919 non-null   object 
     41  CentralAir     2919 non-null   object 
     42  Electrical     2918 non-null   object 
     43  1stFlrSF       2919 non-null   int64  
     44  2ndFlrSF       2919 non-null   int64  
     45  LowQualFinSF   2919 non-null   int64  
     46  GrLivArea      2919 non-null   int64  
     47  BsmtFullBath   2917 non-null   float64
     48  BsmtHalfBath   2917 non-null   float64
     49  FullBath       2919 non-null   int64  
     50  HalfBath       2919 non-null   int64  
     51  BedroomAbvGr   2919 non-null   int64  
     52  KitchenAbvGr   2919 non-null   int64  
     53  KitchenQual    2918 non-null   object 
     54  TotRmsAbvGrd   2919 non-null   int64  
     55  Functional     2917 non-null   object 
     56  Fireplaces     2919 non-null   int64  
     57  FireplaceQu    1499 non-null   object 
     58  GarageType     2762 non-null   object 
     59  GarageYrBlt    2760 non-null   float64
     60  GarageFinish   2760 non-null   object 
     61  GarageCars     2918 non-null   float64
     62  GarageArea     2918 non-null   float64
     63  GarageQual     2760 non-null   object 
     64  GarageCond     2760 non-null   object 
     65  PavedDrive     2919 non-null   object 
     66  WoodDeckSF     2919 non-null   int64  
     67  OpenPorchSF    2919 non-null   int64  
     68  EnclosedPorch  2919 non-null   int64  
     69  3SsnPorch      2919 non-null   int64  
     70  ScreenPorch    2919 non-null   int64  
     71  PoolArea       2919 non-null   int64  
     72  PoolQC         10 non-null     object 
     73  Fence          571 non-null    object 
     74  MiscFeature    105 non-null    object 
     75  MiscVal        2919 non-null   int64  
     76  MoSold         2919 non-null   int64  
     77  YrSold         2919 non-null   int64  
     78  SaleType       2918 non-null   object 
     79  SaleCondition  2919 non-null   object 
     80  SalePrice      2919 non-null   int64  
    dtypes: float64(11), int64(27), object(43)
    memory usage: 1.8+ MB
    

So, now we know that there are different data types and null values in our dataframe. First we need to get rid of null data. For this purposes check for the columns with null values.


```python
# Create function
def where_is_null_values(df):
    
    # Create temporary series with number of null data cells
    temp_df = df.isna().sum()

    # Filter that series and get column's names with number of null cells
    temp_df = temp_df[temp_df > 0]
    
    return temp_df.sort_values(ascending=False)

# Launch our function
where_is_null_values(df)
```




    PoolQC          2909
    MiscFeature     2814
    Alley           2721
    Fence           2348
    FireplaceQu     1420
    LotFrontage      486
    GarageFinish     159
    GarageQual       159
    GarageCond       159
    GarageYrBlt      159
    GarageType       157
    BsmtExposure      82
    BsmtCond          82
    BsmtQual          81
    BsmtFinType2      80
    BsmtFinType1      79
    MasVnrType        24
    MasVnrArea        23
    MSZoning           4
    BsmtFullBath       2
    BsmtHalfBath       2
    Functional         2
    Utilities          2
    GarageArea         1
    GarageCars         1
    Electrical         1
    KitchenQual        1
    TotalBsmtSF        1
    BsmtUnfSF          1
    BsmtFinSF2         1
    BsmtFinSF1         1
    Exterior2nd        1
    Exterior1st        1
    SaleType           1
    dtype: int64



Now we have to examine each column and deside what to do with missing data.
<br>First explore PoolQC


```python
# Get unique values from PoolQC column
print(df['PoolQC'].unique())
```

    [nan 'Ex' 'Fa' 'Gd']
    

Here we can see that we have 3 unique values and nan values.<br>
Go to data description and read 
> PoolQC: Pool quality
> 
>    Ex   Excellent<br>
>    Gd   Good<br>
>    TA   Average/Typical<br>
>    Fa   Fair<br>
>    NA   No Pool 

Now we understand that we need to replace Nan values with No Pool. <br>
And we can do the same with 
<br>Fence - No Fence, 
<br>MiscFeature - None, 
<br>GarageFinish - No Garage
<br>GarageQual - No Garage
<br>GarageCond - No Garage
<br>GarageYrBlt  - No Garage
<br>GarageType - No Garage
<br>BsmtExposure - No Basement
<br>BsmtCond - No Basement
<br>BsmtQual - No Basement
<br>BsmtFinType2 - No Basement
<br>BsmtFinType1 - No Basement
<br>FireplaceQu - No Fireplace
<br>Alley - No alley access
<br>MasVnrType - None 


```python
# Replace those null data 
df['PoolQC'] = df['PoolQC'].fillna('No Pool')
df['Fence'] = df['Fence'].fillna('No Fence') 
df['MiscFeature'] = df['MiscFeature'].fillna('None') 
df['GarageFinish'] = df['GarageFinish'].fillna('No Garage') 
df['GarageQual'] = df['GarageQual'].fillna('No Garage') 
df['GarageCond'] = df['GarageCond'].fillna('No Garage') 
df['GarageYrBlt'] = df['GarageYrBlt'].fillna('No Garage') 
df['GarageType'] = df['GarageType'].fillna('No Garage') 
df['BsmtExposure'] = df['BsmtExposure'].fillna('No') 
df['BsmtCond'] = df['BsmtCond'].fillna('No')
df['BsmtQual'] = df['BsmtQual'].fillna('No')
df['BsmtFinType2'] = df['BsmtExposure'].fillna('No')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('No')
df['FireplaceQu'] = df['FireplaceQu'].fillna('No Fireplace')
df['Alley'] = df['Alley'].fillna('No alley access')
df['MasVnrType'] = df['MasVnrType'].fillna('None')

# Check for null values
where_is_null_values(df)
```




    LotFrontage     486
    MasVnrArea       23
    MSZoning          4
    BsmtFullBath      2
    Utilities         2
    Functional        2
    BsmtHalfBath      2
    GarageArea        1
    GarageCars        1
    KitchenQual       1
    TotalBsmtSF       1
    Electrical        1
    BsmtUnfSF         1
    BsmtFinSF2        1
    BsmtFinSF1        1
    Exterior2nd       1
    Exterior1st       1
    SaleType          1
    dtype: int64



LotFrontage column has numeric values. I'll replace missing values by median LotFrontage of neighborgood houses.


```python
# Replace missing values
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# Check for null values
where_is_null_values(df)
```




    MasVnrArea      23
    MSZoning         4
    BsmtFullBath     2
    Functional       2
    BsmtHalfBath     2
    Utilities        2
    GarageArea       1
    GarageCars       1
    KitchenQual      1
    TotalBsmtSF      1
    Electrical       1
    BsmtUnfSF        1
    BsmtFinSF2       1
    BsmtFinSF1       1
    Exterior2nd      1
    Exterior1st      1
    SaleType         1
    dtype: int64



Much better. Now let's take a look 


```python
# Items that have ['GarageArea'] is nan
df[df['GarageArea'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1116</th>
      <td>2577</td>
      <td>70</td>
      <td>RM</td>
      <td>50.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>MnPrv</td>
      <td>None</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
      <td>WD</td>
      <td>Alloca</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 81 columns</p>
</div>



As we can see the is no garage, so we can fill in Nan values with zero. The same operation with GarageCars


```python
# Replace missed values with 0
df['GarageArea'] = df['GarageArea'].fillna(0)
df['GarageCars'] = df['GarageCars'].fillna(0)

# Check for null values
where_is_null_values(df)
```




    MasVnrArea      23
    MSZoning         4
    Utilities        2
    BsmtFullBath     2
    BsmtHalfBath     2
    Functional       2
    Exterior1st      1
    Exterior2nd      1
    BsmtFinSF1       1
    BsmtFinSF2       1
    BsmtUnfSF        1
    TotalBsmtSF      1
    Electrical       1
    KitchenQual      1
    SaleType         1
    dtype: int64



What about MasVnrArea? 


```python
# Items that have ['MasVnrArea'] is nan
df[df['MasVnrArea'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>234</th>
      <td>235</td>
      <td>60</td>
      <td>RL</td>
      <td>64.0</td>
      <td>7851</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>216500</td>
    </tr>
    <tr>
      <th>529</th>
      <td>530</td>
      <td>20</td>
      <td>RL</td>
      <td>70.0</td>
      <td>32668</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
      <td>WD</td>
      <td>Alloca</td>
      <td>200624</td>
    </tr>
    <tr>
      <th>650</th>
      <td>651</td>
      <td>60</td>
      <td>FV</td>
      <td>65.0</td>
      <td>8125</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>205950</td>
    </tr>
    <tr>
      <th>936</th>
      <td>937</td>
      <td>20</td>
      <td>RL</td>
      <td>67.0</td>
      <td>10083</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>184900</td>
    </tr>
    <tr>
      <th>973</th>
      <td>974</td>
      <td>20</td>
      <td>FV</td>
      <td>95.0</td>
      <td>11639</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>182000</td>
    </tr>
    <tr>
      <th>977</th>
      <td>978</td>
      <td>120</td>
      <td>FV</td>
      <td>35.0</td>
      <td>4274</td>
      <td>Pave</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>11</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>199900</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>1244</td>
      <td>20</td>
      <td>RL</td>
      <td>107.0</td>
      <td>13891</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>9</td>
      <td>2006</td>
      <td>New</td>
      <td>Partial</td>
      <td>465000</td>
    </tr>
    <tr>
      <th>1278</th>
      <td>1279</td>
      <td>60</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9473</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>3</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>237000</td>
    </tr>
    <tr>
      <th>231</th>
      <td>1692</td>
      <td>60</td>
      <td>RL</td>
      <td>64.0</td>
      <td>12891</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>246</th>
      <td>1707</td>
      <td>20</td>
      <td>FV</td>
      <td>90.0</td>
      <td>7993</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>New</td>
      <td>Partial</td>
      <td>0</td>
    </tr>
    <tr>
      <th>422</th>
      <td>1883</td>
      <td>60</td>
      <td>RL</td>
      <td>70.0</td>
      <td>8749</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>11</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>532</th>
      <td>1993</td>
      <td>60</td>
      <td>RL</td>
      <td>64.0</td>
      <td>7750</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>4</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>544</th>
      <td>2005</td>
      <td>20</td>
      <td>RL</td>
      <td>87.0</td>
      <td>10037</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>581</th>
      <td>2042</td>
      <td>60</td>
      <td>FV</td>
      <td>72.5</td>
      <td>7500</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>4</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>851</th>
      <td>2312</td>
      <td>60</td>
      <td>RL</td>
      <td>59.0</td>
      <td>15810</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>0</td>
    </tr>
    <tr>
      <th>865</th>
      <td>2326</td>
      <td>80</td>
      <td>RL</td>
      <td>64.0</td>
      <td>11950</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>10</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>880</th>
      <td>2341</td>
      <td>20</td>
      <td>RL</td>
      <td>85.0</td>
      <td>9965</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>9</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>2350</td>
      <td>60</td>
      <td>FV</td>
      <td>112.0</td>
      <td>12217</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>12</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>0</td>
    </tr>
    <tr>
      <th>908</th>
      <td>2369</td>
      <td>120</td>
      <td>FV</td>
      <td>30.0</td>
      <td>5330</td>
      <td>Pave</td>
      <td>Pave</td>
      <td>IR2</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>2593</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>8298</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>9</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>2658</td>
      <td>60</td>
      <td>RL</td>
      <td>103.0</td>
      <td>12867</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>New</td>
      <td>Partial</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1226</th>
      <td>2687</td>
      <td>20</td>
      <td>RL</td>
      <td>49.0</td>
      <td>15218</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>9</td>
      <td>2006</td>
      <td>New</td>
      <td>Partial</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1402</th>
      <td>2863</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>8050</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>23 rows × 81 columns</p>
</div>



Here we can see that items that have MasVnrArea is Nan also have MasVnrTyme None. So we can replace missing values with zero.


```python
# Replace missed values with 0
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Check for null values
where_is_null_values(df)
```




    MSZoning        4
    Utilities       2
    BsmtFullBath    2
    BsmtHalfBath    2
    Functional      2
    Exterior1st     1
    Exterior2nd     1
    BsmtFinSF1      1
    BsmtFinSF2      1
    BsmtUnfSF       1
    TotalBsmtSF     1
    Electrical      1
    KitchenQual     1
    SaleType        1
    dtype: int64



What about Bsmt's. Lets compouse dataframe with BsmtFullBath is nan


```python
# Show rows where BsmtFullBath has Nan value
df[df['BsmtFullBath'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>660</th>
      <td>2121</td>
      <td>20</td>
      <td>RM</td>
      <td>99.0</td>
      <td>5940</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>MnPrv</td>
      <td>None</td>
      <td>0</td>
      <td>4</td>
      <td>2008</td>
      <td>ConLD</td>
      <td>Abnorml</td>
      <td>0</td>
    </tr>
    <tr>
      <th>728</th>
      <td>2189</td>
      <td>20</td>
      <td>RL</td>
      <td>123.0</td>
      <td>47007</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 81 columns</p>
</div>



As we can see there is no basement in other columns (BsmtQual	BsmtCond	BsmtExposure etc.)


```python
# Replace those NaNs by zero
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)


# Check for null values
where_is_null_values(df)
```




    MSZoning       4
    Utilities      2
    Functional     2
    Exterior1st    1
    Exterior2nd    1
    Electrical     1
    KitchenQual    1
    SaleType       1
    dtype: int64



For other values I'll replace it with most frequent value


```python
# Check for most frequent value (repeat for other)
df['KitchenQual'].value_counts()
```




    TA    1492
    Gd    1151
    Ex     205
    Fa      70
    Name: KitchenQual, dtype: int64




```python
# Replace nan in object and categorical columns
df['KitchenQual'] = df['KitchenQual'].transform(lambda x: x.fillna("TA"))
df['Exterior1st'] = df['Exterior1st'].transform(lambda x: x.fillna("VinylSd"))
df['Exterior2nd'] = df['Exterior2nd'].transform(lambda x: x.fillna("VinylSd"))
df['SaleType'] = df['SaleType'].transform(lambda x: x.fillna("WD"))
df['Functional'] = df['Functional'].transform(lambda x: x.fillna("Typ"))
df['Electrical'] = df['Electrical'].transform(lambda x: x.fillna("SBrkr"))
df['MSZoning'] = df['MSZoning'].transform(lambda x: x.fillna("RL"))
df['Utilities'] = df['Utilities'].transform(lambda x: x.fillna("NA"))

# Check for null values
where_is_null_values(df)
```




    Series([], dtype: int64)



Now we have dataframe without NaN values. Let's return train and test df's again.

Now it's time to split df's


```python
# Split df to train and test (as it was before joining). train df had 1460 rows and test 1459 rows
train = df.iloc[:1460,:]
test  = df.iloc[1460:, :]

# Drop SalePrice column
test.drop('SalePrice', axis=1, inplace=True)
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>No Pool</td>
      <td>MnPrv</td>
      <td>None</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>No Pool</td>
      <td>MnPrv</td>
      <td>None</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>No alley access</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>No Pool</td>
      <td>No Fence</td>
      <td>None</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



# Check for throwouts
We need to exclude throwouts from train dataframe. For this purposes get numeric columns and plot boxplots.


```python
# Get all numeric columns names
numeric_columns = train.describe().columns
```


```python
i=1
plt.figure(figsize=(20,80))
for column in numeric_columns:
    plt.subplot(20, 4, i)
    plt.boxplot(train[column])
    plt.title(column)
    i=i+1 
plt.show() 
```


    
![png](output_43_0.png)
    


Its time to remove data


```python
# Remove rows with throwouts. Add or remove other columns to test how it works with final performance
train = train.drop(train[train['LotFrontage'] > 250].index)
# train = train.drop(train[train['MSSubClass'] > 150].index)
# train = train.drop(train[train['BsmtFinSF1'] > 4500].index)

train.shape
```




    (1458, 81)




```python
def get_dataframe_with_dummy_vars(df, list_of_columns):
    # Input Dataframe and List of the columns that needs to be turned 
    # into dummy variables and get new dataframe 
    
    # For each column in list
    for column in list_of_columns:
        
        # Get dummies dataframe
        dummy_var_df = pd.get_dummies(df[column])
        
        # Create dataframe with names of new columns
        dummy_df = pd.DataFrame(df[column].value_counts())
        dummy_df.reset_index(inplace=True)
        
        # For each name of the column 
        for index, value in enumerate(dummy_df['index']):
            
            # Get name of the var
            name = str(dummy_df.iloc[index,0])
            
            # Create new column name
            new_column_name = column + "_" + name 
            
            # Rename column
            dummy_var_df.rename(columns={name:new_column_name}, inplace=True)
        
        # Merge data frame "df" and "dummy df" 
        df = pd.concat([df, dummy_var_df], axis=1)

        # drop original column "fuel-type" from "df"
        df.drop(column, axis = 1, inplace=True)
        
    return(df)
```


```python
# Append train df with test df
df = train.append(test)

# Check for df's shape
df.shape
```




    (2917, 81)




```python
# Get list of columns that are 'object' or 'category' data type
list_of_cols =  list(df.select_dtypes(['object', 'category']).columns)
print(list_of_cols)

#Get df with dummies
df_d = get_dataframe_with_dummy_vars(df, list_of_cols)
df_d.shape

train_df_lenth = len(train)

# Split df to train and test (as it was before joining). train df had 1460 rows and test 1459 rows
train = df_d.iloc[:train_df_lenth,:]
test  = df_d.iloc[train_df_lenth:, :]

# Drop SalePrice column
test.drop('SalePrice', axis=1, inplace=True)
test.head()
```

    ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>80.0</td>
      <td>11622</td>
      <td>5</td>
      <td>6</td>
      <td>1961</td>
      <td>1961</td>
      <td>0.0</td>
      <td>468.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>81.0</td>
      <td>14267</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>108.0</td>
      <td>923.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>74.0</td>
      <td>13830</td>
      <td>5</td>
      <td>5</td>
      <td>1997</td>
      <td>1998</td>
      <td>0.0</td>
      <td>791.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>78.0</td>
      <td>9978</td>
      <td>6</td>
      <td>6</td>
      <td>1998</td>
      <td>1998</td>
      <td>20.0</td>
      <td>602.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>43.0</td>
      <td>5005</td>
      <td>8</td>
      <td>5</td>
      <td>1992</td>
      <td>1992</td>
      <td>0.0</td>
      <td>263.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 402 columns</p>
</div>



<a id="MLM"></a>
# Machine Learning

Define functions first


```python
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean()
    return rmse

def evaluation(y, predictions):
    "Function Counts evaluation errors"

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared

def run_ml_and_get_performance_results(models_list):
    # Create dataframe
    models = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2 Score", "RMSE (Cross-Validation)"])
    # For each model in list
    for model in models_list:
        print("-"*30)
        str_model = str(model)
        print("Model ", str_model, " Started")
        
        # Check for Linear model
        if "Polynomial" in str_model:
            X_train_d = model.fit_transform(X_train)
            X_test_d = model.transform(X_test)

            lin_reg = LinearRegression()
            lin_reg.fit(X_train_d, y_train)
            predictions = lin_reg.predict(X_test_d)
            # Get coefficients
            mae, mse, rmse, r_squared = evaluation(y_test, predictions)       
            rmse_cross_val = rmse_cv(lin_reg)
            
        else: 
            # Fit model
            model.fit(X_train, y_train)
            # Get predictions on test data
            predictions = model.predict(X_test)
            # Get coefficients
            mae, mse, rmse, r_squared = evaluation(y_test, predictions)       
            rmse_cross_val = rmse_cv(model)
        
        print("MAE:", mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2 Score:", r_squared)
        print("-"*30)
        print("RMSE Cross-Validation:", rmse_cross_val)
        
        # Add new row to df
        new_row = {"Model": str_model[0:10],"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
        models = models.append(new_row, ignore_index=True)
    
    return models
```


```python
# Set X and y 
y = train[['SalePrice']]
train.drop('SalePrice', axis=1, inplace=True)
X = train
```


```python
X.dtypes.T
```




    Id                         int64
    MSSubClass                 int64
    LotFrontage              float64
    LotArea                    int64
    OverallQual                int64
                              ...   
    SaleCondition_AdjLand      uint8
    SaleCondition_Alloca       uint8
    SaleCondition_Family       uint8
    SaleCondition_Normal       uint8
    SaleCondition_Partial      uint8
    Length: 402, dtype: object




```python
# Import lib
from sklearn.preprocessing import StandardScaler

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X
```




    array([[-1.72964085,  0.07279605, -0.2355964 , ..., -0.11793306,
             0.46695301, -0.3048828 ],
           [-1.72726813, -0.872742  ,  0.49406349, ..., -0.11793306,
             0.46695301, -0.3048828 ],
           [-1.72489541,  0.07279605, -0.08966442, ..., -0.11793306,
             0.46695301, -0.3048828 ],
           ...,
           [ 1.72741134,  0.30918057, -0.18695241, ..., -0.11793306,
             0.46695301, -0.3048828 ],
           [ 1.72978406, -0.872742  , -0.08966442, ..., -0.11793306,
             0.46695301, -0.3048828 ],
           [ 1.73215678, -0.872742  ,  0.25084352, ..., -0.11793306,
             0.46695301, -0.3048828 ]])




```python
# Import lib
from sklearn.model_selection import train_test_split, cross_val_score

# Split the data into train and test chunks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# Import requred libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error

# Create List for tests
models_list = [Ridge(), 
               Lasso(),
               ElasticNet(),
               SVR(C=100000),
               RandomForestRegressor(n_estimators=100),
               XGBRegressor(n_estimators=1000, learning_rate=0.01),
               PolynomialFeatures(degree=2)
               ]


```


```python
# Uncomment this two lines to see resul of function's work

# results = run_ml_and_get_performance_results(models_list)
# results.sort_values(by="RMSE (Cross-Validation)")
```

# Price prediction
When we choosed the model, let's predict prices. Start from dataframe preparation


```python
# Set X_test
X_train = X
X_test = test

X_test = scaler.fit_transform(X_test)
X_train = scaler.fit_transform(X_train)
```


```python
# Set the model
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01)

# Fit the model on all train data (before split, U can try to train it on X_train, y_train)
xgb.fit(X_train, y) 
```




    XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
                 colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                 early_stopping_rounds=None, enable_categorical=False,
                 eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                 importance_type=None, interaction_constraints='',
                 learning_rate=0.01, max_bin=256, max_cat_to_onehot=4,
                 max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                 missing=nan, monotone_constraints='()', n_estimators=1000,
                 n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
                 reg_alpha=0, reg_lambda=1, ...)




```python
# Make prediction
predictions = xgb.predict(X_test).round()

# Export results
Submission = pd.DataFrame({ 'Id': test['Id'], 'SalePrice': predictions })
```


```python
Submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>128552.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>157319.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>192200.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>191158.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>187674.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Export csv file
Submission.to_csv('submission.csv', index=False)
```


```python

```
