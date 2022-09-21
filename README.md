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
import pandas as pd
import numpy as np
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
train.head()
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
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



Explore dataframe


```python
train.shape
```




    (1460, 81)




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB


Visualize correlation between numerical variables in "train" dataframe


```python
plt.figure(figsize=(10,10))
sns.heatmap(train.corr(), cmap="RdBu")
plt.title("Correlations Between Variables", size=15)
plt.show()
```


    
![png](output_12_0.png)
    


Create new dataframe using only columns which correlates to SalePrice


```python
# Create list of important columns
important_num_cols = list(train.corr()["SalePrice"][(train.corr()["SalePrice"]>0.30) | (train.corr()["SalePrice"]<-0.30)].index)

# Create list of Object type columns
cat_cols = ["MSZoning", "Utilities","BldgType","Heating","KitchenQual","SaleCondition","LandSlope"]

# Create dataframe
important_cols = important_num_cols + cat_cols
train_df = train[important_cols]

train_df.head()
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
      <th>LotFrontage</th>
      <th>OverallQual</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>GrLivArea</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>SalePrice</th>
      <th>MSZoning</th>
      <th>Utilities</th>
      <th>BldgType</th>
      <th>Heating</th>
      <th>KitchenQual</th>
      <th>SaleCondition</th>
      <th>LandSlope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.0</td>
      <td>7</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>856</td>
      <td>856</td>
      <td>854</td>
      <td>1710</td>
      <td>...</td>
      <td>0</td>
      <td>61</td>
      <td>208500</td>
      <td>RL</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Normal</td>
      <td>Gtl</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80.0</td>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>1262</td>
      <td>1262</td>
      <td>0</td>
      <td>1262</td>
      <td>...</td>
      <td>298</td>
      <td>0</td>
      <td>181500</td>
      <td>RL</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Normal</td>
      <td>Gtl</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68.0</td>
      <td>7</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>920</td>
      <td>920</td>
      <td>866</td>
      <td>1786</td>
      <td>...</td>
      <td>0</td>
      <td>42</td>
      <td>223500</td>
      <td>RL</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Normal</td>
      <td>Gtl</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.0</td>
      <td>7</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>756</td>
      <td>961</td>
      <td>756</td>
      <td>1717</td>
      <td>...</td>
      <td>0</td>
      <td>35</td>
      <td>140000</td>
      <td>RL</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Abnorml</td>
      <td>Gtl</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84.0</td>
      <td>8</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>1145</td>
      <td>1145</td>
      <td>1053</td>
      <td>2198</td>
      <td>...</td>
      <td>192</td>
      <td>84</td>
      <td>250000</td>
      <td>RL</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Normal</td>
      <td>Gtl</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
#Check for missing values
print(train_df.isna().sum())

print("Total: ",train_df.isna().sum().sum())
```

    LotFrontage      259
    OverallQual        0
    YearBuilt          0
    YearRemodAdd       0
    MasVnrArea         8
    BsmtFinSF1         0
    TotalBsmtSF        0
    1stFlrSF           0
    2ndFlrSF           0
    GrLivArea          0
    FullBath           0
    TotRmsAbvGrd       0
    Fireplaces         0
    GarageYrBlt       81
    GarageCars         0
    GarageArea         0
    WoodDeckSF         0
    OpenPorchSF        0
    SalePrice          0
    MSZoning           0
    Utilities          0
    BldgType           0
    Heating            0
    KitchenQual        0
    SaleCondition      0
    LandSlope          0
    dtype: int64
    Total:  348


We have missing values in 3 columns. Let's explore them deeply.


```python
sns.scatterplot(x="SalePrice", y="GarageYrBlt", data=train_df)
```




    <AxesSubplot:xlabel='SalePrice', ylabel='GarageYrBlt'>




    
![png](output_17_1.png)
    



```python
train["GarageYrBlt"].describe()
```




    count    1379.000000
    mean     1978.506164
    std        24.689725
    min      1900.000000
    25%      1961.000000
    50%      1980.000000
    75%      2002.000000
    max      2010.000000
    Name: GarageYrBlt, dtype: float64



GarageYrBlt column contains list of years when Garage was built. Let's cut it to 5 bins.


```python
# Convert data from Categorical to numeric type
train_df['GarageYrBlt']=pd.to_numeric(train_df['GarageYrBlt'])

# Replace Nan by '1'
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(1)

# Define bins borders
bins = [0, 1900, 1950, 1970, 1990, 2010]

# Create list of bined values
group_names = ['No_data', '1900-1949', '1950-1969', '1970-1989', '1990-2010']

# Create new column to DF
train_df['GarageYrBlt'] = pd.cut(train_df['GarageYrBlt'], bins, labels=group_names, include_lowest=True)

train_df['GarageYrBlt'].value_counts()
```




    1990-2010    582
    1950-1969    332
    1970-1989    277
    1900-1949    187
    No_data       82
    Name: GarageYrBlt, dtype: int64




```python
# Check for missing values again
train_df.isna().sum()
```




    LotFrontage      259
    OverallQual        0
    YearBuilt          0
    YearRemodAdd       0
    MasVnrArea         8
    BsmtFinSF1         0
    TotalBsmtSF        0
    1stFlrSF           0
    2ndFlrSF           0
    GrLivArea          0
    FullBath           0
    TotRmsAbvGrd       0
    Fireplaces         0
    GarageYrBlt        0
    GarageCars         0
    GarageArea         0
    WoodDeckSF         0
    OpenPorchSF        0
    SalePrice          0
    MSZoning           0
    Utilities          0
    BldgType           0
    Heating            0
    KitchenQual        0
    SaleCondition      0
    LandSlope          0
    dtype: int64



Visualize how binned data looks now.


```python
# Plot barplot
sns.barplot(x=train_df['GarageYrBlt'], y=train_df['SalePrice'], data=train_df)
```




    <AxesSubplot:xlabel='GarageYrBlt', ylabel='SalePrice'>




    
![png](output_23_1.png)
    


LotFrontage column has numeric values. I'll replace missing values by median LotFrontage of neighborgood houses.


```python
# Replace missing values
train_df["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
train_df["LotFrontage"].isna().sum()
```




    0




```python
train_df.isna().sum()
```




    LotFrontage      0
    OverallQual      0
    YearBuilt        0
    YearRemodAdd     0
    MasVnrArea       8
    BsmtFinSF1       0
    TotalBsmtSF      0
    1stFlrSF         0
    2ndFlrSF         0
    GrLivArea        0
    FullBath         0
    TotRmsAbvGrd     0
    Fireplaces       0
    GarageYrBlt      0
    GarageCars       0
    GarageArea       0
    WoodDeckSF       0
    OpenPorchSF      0
    SalePrice        0
    MSZoning         0
    Utilities        0
    BldgType         0
    Heating          0
    KitchenQual      0
    SaleCondition    0
    LandSlope        0
    dtype: int64



Explore MasVnrArea column


```python
train_df["MasVnrArea"].describe()
```




    count    1452.000000
    mean      103.685262
    std       181.066207
    min         0.000000
    25%         0.000000
    50%         0.000000
    75%       166.000000
    max      1600.000000
    Name: MasVnrArea, dtype: float64




```python
train_df.dtypes
```




    LotFrontage       float64
    OverallQual         int64
    YearBuilt           int64
    YearRemodAdd        int64
    MasVnrArea        float64
    BsmtFinSF1          int64
    TotalBsmtSF         int64
    1stFlrSF            int64
    2ndFlrSF            int64
    GrLivArea           int64
    FullBath            int64
    TotRmsAbvGrd        int64
    Fireplaces          int64
    GarageYrBlt      category
    GarageCars          int64
    GarageArea          int64
    WoodDeckSF          int64
    OpenPorchSF         int64
    SalePrice           int64
    MSZoning           object
    Utilities          object
    BldgType           object
    Heating            object
    KitchenQual        object
    SaleCondition      object
    LandSlope          object
    dtype: object




```python
sns.scatterplot(x=train_df["MasVnrArea"], y=train_df["SalePrice"], data=train_df)
```




    <AxesSubplot:xlabel='MasVnrArea', ylabel='SalePrice'>




    
![png](output_30_1.png)
    


We can see lots of null values and linear depency. Let's cut the column to 8 pieces


```python
# Define bins
bins = np.linspace(min(train_df["MasVnrArea"]), max(train_df["MasVnrArea"]), 9)

# Set group names
g_names = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000-1200', '1200-1400', '1400-1600']

# Cut it
train_df["MasVnrArea"] = pd.cut(train_df["MasVnrArea"], bins, labels=g_names, include_lowest=True )

train_df["MasVnrArea"].value_counts()
```




    0-200        1154
    200-400       198
    400-600        61
    600-800        25
    800-1000        7
    1000-1200       5
    1200-1400       1
    1400-1600       1
    Name: MasVnrArea, dtype: int64




```python
# Plot barplot
sns.barplot(x=train_df["MasVnrArea"], y=train_df["SalePrice"], data=train_df)
```




    <AxesSubplot:xlabel='MasVnrArea', ylabel='SalePrice'>




    
![png](output_33_1.png)
    


Check for vars that strongly correlates to each other


```python
def get_vars_list_with_strong_correlation(dataframe):
    # Function returns list of vars that correlates to each  other with value of correlation
    # coefficien more then 90%
    
    # Create an empty list
    CorField = []
    
    # Count correlation coefficients
    CorrKoef=dataframe.corr()    
    
    # For each column in df
    for column_index in CorrKoef:
        
        # For each var index in filtered df where value more then 90%
        for var_index in CorrKoef.index[CorrKoef[column_index] > 0.9]:
            
            # Check if var index already in list and check for coefficient 
            # for same variable
            if column_index != var_index and var_index not in CorField and column_index not in CorField:
                CorField.append(var_index)
                print ("%s-->%s: r^2=%f" % (column_index,var_index, CorrKoef[column_index][CorrKoef.index==var_index].values[0]))
    return(CorField)   
```


```python
# Run function and get the list of vars
get_vars_list_with_strong_correlation(train_df)
```




    []



As we got empty list it means that there is no such vars.


```python
# Plot the charts
sns.pairplot(train_df[important_cols])
```




    <seaborn.axisgrid.PairGrid at 0x7fc5c7c093d0>




    
![png](output_38_1.png)
    


Transform dataframe to df with dummies


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
            name = dummy_df.iloc[index,0]
            
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
# Get list of columns that are 'object' or 'category' data type
list_of_cols =  list(train_df.select_dtypes(['object', 'category']).columns)
print(list_of_cols)

#Get df with dummies
train_df_d = get_dataframe_with_dummy_vars(train_df, list_of_cols)
```

    ['MasVnrArea', 'GarageYrBlt', 'MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']



```python
train_df_d.head().T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LotFrontage</th>
      <td>65.0</td>
      <td>80.0</td>
      <td>68.0</td>
      <td>60.0</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>7.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>2003.0</td>
      <td>1976.0</td>
      <td>2001.0</td>
      <td>1915.0</td>
      <td>2000.0</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>2003.0</td>
      <td>1976.0</td>
      <td>2002.0</td>
      <td>1970.0</td>
      <td>2000.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>706.0</td>
      <td>978.0</td>
      <td>486.0</td>
      <td>216.0</td>
      <td>655.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>SaleCondition_Normal</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>SaleCondition_Partial</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>LandSlope_Gtl</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>LandSlope_Mod</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>LandSlope_Sev</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 5 columns</p>
</div>



<a id="ds"></a>
# Data Standartisation

Split the data into X and y chunks


```python
# Set X
X = train_df_d.drop("SalePrice", axis=1)

# Set y
y = train_df_d["SalePrice"]
```

Standardize only numeric values. First we need to remove 3 columns from our list


```python
important_num_cols.remove("SalePrice")
important_num_cols.remove("MasVnrArea")
important_num_cols.remove("GarageYrBlt")
```


```python
# Import lib
from sklearn.preprocessing import StandardScaler

# Normalize data
scaler = StandardScaler()
X[important_num_cols] = scaler.fit_transform(X[important_num_cols])
```


```python
X.head()
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
      <th>LotFrontage</th>
      <th>OverallQual</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>BsmtFinSF1</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>...</th>
      <th>KitchenQual_TA</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
      <th>LandSlope_Gtl</th>
      <th>LandSlope_Mod</th>
      <th>LandSlope_Sev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.231877</td>
      <td>0.651479</td>
      <td>1.050994</td>
      <td>0.878668</td>
      <td>0.575425</td>
      <td>-0.459303</td>
      <td>-0.793434</td>
      <td>1.161852</td>
      <td>0.370333</td>
      <td>0.789741</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.437043</td>
      <td>-0.071836</td>
      <td>0.156734</td>
      <td>-0.429577</td>
      <td>1.171992</td>
      <td>0.466465</td>
      <td>0.257140</td>
      <td>-0.795163</td>
      <td>-0.482512</td>
      <td>0.789741</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.098093</td>
      <td>0.651479</td>
      <td>0.984752</td>
      <td>0.830215</td>
      <td>0.092907</td>
      <td>-0.313369</td>
      <td>-0.627826</td>
      <td>1.189351</td>
      <td>0.515013</td>
      <td>0.789741</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.454850</td>
      <td>0.651479</td>
      <td>-1.863632</td>
      <td>-0.720298</td>
      <td>-0.499274</td>
      <td>-0.687324</td>
      <td>-0.521734</td>
      <td>0.937276</td>
      <td>0.383659</td>
      <td>-1.026041</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.615421</td>
      <td>1.374795</td>
      <td>0.951632</td>
      <td>0.733308</td>
      <td>0.463568</td>
      <td>0.199680</td>
      <td>-0.045611</td>
      <td>1.617877</td>
      <td>1.299326</td>
      <td>0.789741</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>



Train test split


```python
# Import lib
from sklearn.model_selection import train_test_split, cross_val_score

# Split the data into train and test chunks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<a id="MLM"></a>
# Machine learing models

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

results = run_ml_and_get_performance_results(models_list)
```

    ------------------------------
    Model  Ridge()  Started
    MAE: 22606.059934893816
    MSE: 1309278571.2651732
    RMSE: 36183.95461064439
    R2 Score: 0.8293060117784333
    ------------------------------
    RMSE Cross-Validation: 35614.419237946626
    ------------------------------
    Model  Lasso()  Started
    MAE: 22862.581490394266
    MSE: 1327001887.00181
    RMSE: 36428.03710058792
    R2 Score: 0.8269953778812686
    ------------------------------
    RMSE Cross-Validation: 36038.320454781955
    ------------------------------
    Model  ElasticNet()  Started
    MAE: 22297.27253344446
    MSE: 1536211772.7639716
    RMSE: 39194.5375373147
    R2 Score: 0.7997201512336548
    ------------------------------
    RMSE Cross-Validation: 36832.93595373031
    ------------------------------
    Model  SVR(C=100000)  Started
    MAE: 17237.29201307822
    MSE: 1059642041.1924161
    RMSE: 32552.143419326723
    R2 Score: 0.8618517632014754
    ------------------------------
    RMSE Cross-Validation: 29821.97843720208
    ------------------------------
    Model  RandomForestRegressor()  Started
    MAE: 17881.772534246575
    MSE: 782016415.0401495
    RMSE: 27964.55640699758
    R2 Score: 0.898046524500171
    ------------------------------
    RMSE Cross-Validation: 29926.118504806538
    ------------------------------
    Model  XGBRegressor(base_score=None, booster=None, callbacks=None,
                 colsample_bylevel=None, colsample_bynode=None,
                 colsample_bytree=None, early_stopping_rounds=None,
                 enable_categorical=False, eval_metric=None, gamma=None,
                 gpu_id=None, grow_policy=None, importance_type=None,
                 interaction_constraints=None, learning_rate=0.01, max_bin=None,
                 max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
                 max_leaves=None, min_child_weight=None, missing=nan,
                 monotone_constraints=None, n_estimators=1000, n_jobs=None,
                 num_parallel_tree=None, predictor=None, random_state=None,
                 reg_alpha=None, reg_lambda=None, ...)  Started
    MAE: 17404.034567636987
    MSE: 853374024.3337017
    RMSE: 29212.56620589334
    R2 Score: 0.8887434508933809
    ------------------------------
    RMSE Cross-Validation: 28325.729892151274
    ------------------------------
    Model  PolynomialFeatures()  Started
    MAE: 646250074210992.0
    MSE: 1.2300893781672986e+31
    RMSE: 3507263004348688.0
    R2 Score: -1.6036989105035716e+21
    ------------------------------
    RMSE Cross-Validation: 20673685755101.43



```python
results.sort_values(by="RMSE (Cross-Validation)")
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
      <th>Model</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>RMSE</th>
      <th>R2 Score</th>
      <th>RMSE (Cross-Validation)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>XGBRegress</td>
      <td>1.740403e+04</td>
      <td>8.533740e+08</td>
      <td>2.921257e+04</td>
      <td>8.887435e-01</td>
      <td>2.832573e+04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVR(C=1000</td>
      <td>1.723729e+04</td>
      <td>1.059642e+09</td>
      <td>3.255214e+04</td>
      <td>8.618518e-01</td>
      <td>2.982198e+04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RandomFore</td>
      <td>1.788177e+04</td>
      <td>7.820164e+08</td>
      <td>2.796456e+04</td>
      <td>8.980465e-01</td>
      <td>2.992612e+04</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Ridge()</td>
      <td>2.260606e+04</td>
      <td>1.309279e+09</td>
      <td>3.618395e+04</td>
      <td>8.293060e-01</td>
      <td>3.561442e+04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lasso()</td>
      <td>2.286258e+04</td>
      <td>1.327002e+09</td>
      <td>3.642804e+04</td>
      <td>8.269954e-01</td>
      <td>3.603832e+04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ElasticNet</td>
      <td>2.229727e+04</td>
      <td>1.536212e+09</td>
      <td>3.919454e+04</td>
      <td>7.997202e-01</td>
      <td>3.683294e+04</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Polynomial</td>
      <td>6.462501e+14</td>
      <td>1.230089e+31</td>
      <td>3.507263e+15</td>
      <td>-1.603699e+21</td>
      <td>2.067369e+13</td>
    </tr>
  </tbody>
</table>
</div>



XGBRegress shows better result with this settings. You can tune up setting for each model in list upper or add your models when run function. Do not forget import required libs.


```python
# Plot chart if U need
#plt.figure(figsize=(12,8))
#sns.barplot(x=results["Model"], y=results["RMSE (Cross-Validation)"])
#plt.title("Models comparison")
#plt.show
```

# Price prediction
When we choosed the model, let's predict prices. Start from dataframe preparation


```python
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 80 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1459 non-null   int64  
     1   MSSubClass     1459 non-null   int64  
     2   MSZoning       1455 non-null   object 
     3   LotFrontage    1232 non-null   float64
     4   LotArea        1459 non-null   int64  
     5   Street         1459 non-null   object 
     6   Alley          107 non-null    object 
     7   LotShape       1459 non-null   object 
     8   LandContour    1459 non-null   object 
     9   Utilities      1457 non-null   object 
     10  LotConfig      1459 non-null   object 
     11  LandSlope      1459 non-null   object 
     12  Neighborhood   1459 non-null   object 
     13  Condition1     1459 non-null   object 
     14  Condition2     1459 non-null   object 
     15  BldgType       1459 non-null   object 
     16  HouseStyle     1459 non-null   object 
     17  OverallQual    1459 non-null   int64  
     18  OverallCond    1459 non-null   int64  
     19  YearBuilt      1459 non-null   int64  
     20  YearRemodAdd   1459 non-null   int64  
     21  RoofStyle      1459 non-null   object 
     22  RoofMatl       1459 non-null   object 
     23  Exterior1st    1458 non-null   object 
     24  Exterior2nd    1458 non-null   object 
     25  MasVnrType     1443 non-null   object 
     26  MasVnrArea     1444 non-null   float64
     27  ExterQual      1459 non-null   object 
     28  ExterCond      1459 non-null   object 
     29  Foundation     1459 non-null   object 
     30  BsmtQual       1415 non-null   object 
     31  BsmtCond       1414 non-null   object 
     32  BsmtExposure   1415 non-null   object 
     33  BsmtFinType1   1417 non-null   object 
     34  BsmtFinSF1     1458 non-null   float64
     35  BsmtFinType2   1417 non-null   object 
     36  BsmtFinSF2     1458 non-null   float64
     37  BsmtUnfSF      1458 non-null   float64
     38  TotalBsmtSF    1458 non-null   float64
     39  Heating        1459 non-null   object 
     40  HeatingQC      1459 non-null   object 
     41  CentralAir     1459 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1459 non-null   int64  
     44  2ndFlrSF       1459 non-null   int64  
     45  LowQualFinSF   1459 non-null   int64  
     46  GrLivArea      1459 non-null   int64  
     47  BsmtFullBath   1457 non-null   float64
     48  BsmtHalfBath   1457 non-null   float64
     49  FullBath       1459 non-null   int64  
     50  HalfBath       1459 non-null   int64  
     51  BedroomAbvGr   1459 non-null   int64  
     52  KitchenAbvGr   1459 non-null   int64  
     53  KitchenQual    1458 non-null   object 
     54  TotRmsAbvGrd   1459 non-null   int64  
     55  Functional     1457 non-null   object 
     56  Fireplaces     1459 non-null   int64  
     57  FireplaceQu    729 non-null    object 
     58  GarageType     1383 non-null   object 
     59  GarageYrBlt    1381 non-null   float64
     60  GarageFinish   1381 non-null   object 
     61  GarageCars     1458 non-null   float64
     62  GarageArea     1458 non-null   float64
     63  GarageQual     1381 non-null   object 
     64  GarageCond     1381 non-null   object 
     65  PavedDrive     1459 non-null   object 
     66  WoodDeckSF     1459 non-null   int64  
     67  OpenPorchSF    1459 non-null   int64  
     68  EnclosedPorch  1459 non-null   int64  
     69  3SsnPorch      1459 non-null   int64  
     70  ScreenPorch    1459 non-null   int64  
     71  PoolArea       1459 non-null   int64  
     72  PoolQC         3 non-null      object 
     73  Fence          290 non-null    object 
     74  MiscFeature    51 non-null     object 
     75  MiscVal        1459 non-null   int64  
     76  MoSold         1459 non-null   int64  
     77  YrSold         1459 non-null   int64  
     78  SaleType       1458 non-null   object 
     79  SaleCondition  1459 non-null   object 
    dtypes: float64(11), int64(26), object(43)
    memory usage: 912.0+ KB



```python
# Create test dataframe with needed columns
important_cols = important_num_cols + cat_cols + ["GarageYrBlt", "MasVnrArea"]
test_df = test[important_cols]
test_df.head()
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
      <th>LotFrontage</th>
      <th>OverallQual</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>BsmtFinSF1</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>...</th>
      <th>OpenPorchSF</th>
      <th>MSZoning</th>
      <th>Utilities</th>
      <th>BldgType</th>
      <th>Heating</th>
      <th>KitchenQual</th>
      <th>SaleCondition</th>
      <th>LandSlope</th>
      <th>GarageYrBlt</th>
      <th>MasVnrArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80.0</td>
      <td>5</td>
      <td>1961</td>
      <td>1961</td>
      <td>468.0</td>
      <td>882.0</td>
      <td>896</td>
      <td>0</td>
      <td>896</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>RH</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Normal</td>
      <td>Gtl</td>
      <td>1961.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>81.0</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>923.0</td>
      <td>1329.0</td>
      <td>1329</td>
      <td>0</td>
      <td>1329</td>
      <td>1</td>
      <td>...</td>
      <td>36</td>
      <td>RL</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Normal</td>
      <td>Gtl</td>
      <td>1958.0</td>
      <td>108.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>74.0</td>
      <td>5</td>
      <td>1997</td>
      <td>1998</td>
      <td>791.0</td>
      <td>928.0</td>
      <td>928</td>
      <td>701</td>
      <td>1629</td>
      <td>2</td>
      <td>...</td>
      <td>34</td>
      <td>RL</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Normal</td>
      <td>Gtl</td>
      <td>1997.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78.0</td>
      <td>6</td>
      <td>1998</td>
      <td>1998</td>
      <td>602.0</td>
      <td>926.0</td>
      <td>926</td>
      <td>678</td>
      <td>1604</td>
      <td>2</td>
      <td>...</td>
      <td>36</td>
      <td>RL</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Normal</td>
      <td>Gtl</td>
      <td>1998.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43.0</td>
      <td>8</td>
      <td>1992</td>
      <td>1992</td>
      <td>263.0</td>
      <td>1280.0</td>
      <td>1280</td>
      <td>0</td>
      <td>1280</td>
      <td>2</td>
      <td>...</td>
      <td>82</td>
      <td>RL</td>
      <td>AllPub</td>
      <td>TwnhsE</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Normal</td>
      <td>Gtl</td>
      <td>1992.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
# Convert data from Categorical to numeric type
test_df['GarageYrBlt']=pd.to_numeric(test_df['GarageYrBlt'])

# Replace Nan by '1'
test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna(1)

# Define bins borders
bins = [0, 1900, 1950, 1970, 1990, 2010]

# Create list of bined values
group_names = ['No_data', '1900-1949', '1950-1969', '1970-1989', '1990-2010']

# Create new column to DF
test_df['GarageYrBlt'] = pd.cut(test_df['GarageYrBlt'], bins, labels=group_names, include_lowest=True)

test_df['GarageYrBlt'].value_counts()
```




    1990-2010    588
    1950-1969    331
    1970-1989    257
    1900-1949    197
    No_data       85
    Name: GarageYrBlt, dtype: int64




```python
test_df['GarageYrBlt'].isna().sum()
```




    1




```python
test_df["MasVnrArea"].describe()
```




    count    1444.000000
    mean      100.709141
    std       177.625900
    min         0.000000
    25%         0.000000
    50%         0.000000
    75%       164.000000
    max      1290.000000
    Name: MasVnrArea, dtype: float64




```python
# Define bins
bins = np.linspace(0, 1600, 9)

# Set group names
g_names = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000-1200', '1200-1400', '1400-1600']

test_df["MasVnrArea"] = pd.cut(test_df["MasVnrArea"], bins, labels=g_names, include_lowest=True )

test_df["MasVnrArea"].value_counts()
```




    0-200        1161
    200-400       176
    400-600        73
    600-800        22
    800-1000        5
    1000-1200       4
    1200-1400       3
    1400-1600       0
    Name: MasVnrArea, dtype: int64




```python
test_df.shape
```




    (1459, 25)




```python
test_df.isna().sum()
```




    LotFrontage      227
    OverallQual        0
    YearBuilt          0
    YearRemodAdd       0
    BsmtFinSF1         1
    TotalBsmtSF        1
    1stFlrSF           0
    2ndFlrSF           0
    GrLivArea          0
    FullBath           0
    TotRmsAbvGrd       0
    Fireplaces         0
    GarageCars         1
    GarageArea         1
    WoodDeckSF         0
    OpenPorchSF        0
    MSZoning           4
    Utilities          2
    BldgType           0
    Heating            0
    KitchenQual        1
    SaleCondition      0
    LandSlope          0
    GarageYrBlt        1
    MasVnrArea        15
    dtype: int64



Replace missing values


```python
# Replace numeric column's Nan values by mean
test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].transform(lambda x: x.fillna(x.median()))
test_df['GarageCars'] = test_df['GarageCars'].transform(lambda x: x.fillna(x.median()))
test_df['GarageArea'] = test_df['GarageArea'].transform(lambda x: x.fillna(x.median()))
test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].transform(lambda x: x.fillna(x.median()))
test_df["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# Replace nan in object and categorical columns
test_df["MasVnrArea"] = test_df["MasVnrArea"].transform(lambda x: x.fillna("0-200"))
test_df['KitchenQual'] = test_df['KitchenQual'].transform(lambda x: x.fillna("TA"))
test_df['MSZoning'] = test_df['MSZoning'].transform(lambda x: x.fillna("RL"))
test_df['Utilities'] = test_df['Utilities'].transform(lambda x: x.fillna("NA"))
test_df['GarageYrBlt'] = test_df['GarageYrBlt'].transform(lambda x: x.fillna("No_data"))
```


```python
# Check for most frequent value in column (helps create list upper for each column)
test_df["GarageYrBlt"].value_counts()
```




    1990-2010    588
    1950-1969    331
    1970-1989    257
    1900-1949    197
    No_data       86
    Name: GarageYrBlt, dtype: int64




```python
#Check for missing values
print(test_df.isna().sum())

print("Total: ",test_df.isna().sum().sum())
```

    LotFrontage      0
    OverallQual      0
    YearBuilt        0
    YearRemodAdd     0
    BsmtFinSF1       0
    TotalBsmtSF      0
    1stFlrSF         0
    2ndFlrSF         0
    GrLivArea        0
    FullBath         0
    TotRmsAbvGrd     0
    Fireplaces       0
    GarageCars       0
    GarageArea       0
    WoodDeckSF       0
    OpenPorchSF      0
    MSZoning         0
    Utilities        0
    BldgType         0
    Heating          0
    KitchenQual      0
    SaleCondition    0
    LandSlope        0
    GarageYrBlt      0
    MasVnrArea       0
    dtype: int64
    Total:  0


Create new df with dummies


```python
# Get list of columns that are 'object' data type
list_of_cols =  list(test_df.select_dtypes(['object']).columns)
print(list_of_cols)

#Get df with dummies
test_df_d = get_dataframe_with_dummy_vars(test_df, list_of_cols)
```

    ['MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']



```python
test_df_d.head().T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LotFrontage</th>
      <td>80.0</td>
      <td>81.0</td>
      <td>74.0</td>
      <td>78.0</td>
      <td>43.0</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>1961</td>
      <td>1958</td>
      <td>1997</td>
      <td>1998</td>
      <td>1992</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>1961</td>
      <td>1958</td>
      <td>1998</td>
      <td>1998</td>
      <td>1992</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>468.0</td>
      <td>923.0</td>
      <td>791.0</td>
      <td>602.0</td>
      <td>263.0</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>882.0</td>
      <td>1329.0</td>
      <td>928.0</td>
      <td>926.0</td>
      <td>1280.0</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>896</td>
      <td>1329</td>
      <td>928</td>
      <td>926</td>
      <td>1280</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>0</td>
      <td>0</td>
      <td>701</td>
      <td>678</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>896</td>
      <td>1329</td>
      <td>1629</td>
      <td>1604</td>
      <td>1280</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>5</td>
      <td>6</td>
      <td>6</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>730.0</td>
      <td>312.0</td>
      <td>482.0</td>
      <td>470.0</td>
      <td>506.0</td>
    </tr>
    <tr>
      <th>WoodDeckSF</th>
      <td>140</td>
      <td>393</td>
      <td>212</td>
      <td>360</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>0</td>
      <td>36</td>
      <td>34</td>
      <td>36</td>
      <td>82</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>1950-1969</td>
      <td>1950-1969</td>
      <td>1990-2010</td>
      <td>1990-2010</td>
      <td>1990-2010</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0-200</td>
      <td>0-200</td>
      <td>0-200</td>
      <td>0-200</td>
      <td>0-200</td>
    </tr>
    <tr>
      <th>MSZoning_C (all)</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MSZoning_FV</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MSZoning_RH</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MSZoning_RL</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MSZoning_RM</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Utilities_AllPub</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Utilities_NA</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BldgType_1Fam</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BldgType_2fmCon</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BldgType_Duplex</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BldgType_Twnhs</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BldgType_TwnhsE</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Heating_GasA</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Heating_GasW</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Heating_Grav</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Heating_Wall</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>KitchenQual_Ex</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>KitchenQual_Fa</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>KitchenQual_Gd</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>KitchenQual_TA</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SaleCondition_Abnorml</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SaleCondition_AdjLand</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SaleCondition_Alloca</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SaleCondition_Family</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SaleCondition_Normal</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SaleCondition_Partial</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LandSlope_Gtl</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LandSlope_Mod</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LandSlope_Sev</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#train_df_d.shape
test_df_d.shape
```




    (1459, 47)



It looks like in test df we have less colums. Check for mismatches.


```python
#Check for missing columns
train_cols = train_df_d.columns
test_cols = test_df_d.columns

common_cols = train_cols.intersection(test_cols)
train_not_test = train_cols.difference(test_cols)
test_not_train = test_cols.difference(train_cols)

print("Train not test", train_not_test)
print("Test not train", test_not_train)
```

    Train not test Index(['GarageYrBlt_1900-1949', 'GarageYrBlt_1950-1969',
           'GarageYrBlt_1970-1989', 'GarageYrBlt_1990-2010', 'GarageYrBlt_No_data',
           'Heating_Floor', 'Heating_OthW', 'MasVnrArea_0-200',
           'MasVnrArea_1000-1200', 'MasVnrArea_1200-1400', 'MasVnrArea_1400-1600',
           'MasVnrArea_200-400', 'MasVnrArea_400-600', 'MasVnrArea_600-800',
           'MasVnrArea_800-1000', 'SalePrice', 'Utilities_NoSeWa'],
          dtype='object')
    Test not train Index(['GarageYrBlt', 'MasVnrArea', 'Utilities_NA'], dtype='object')



```python
# Add missing columns with 0 value
test_df_d[['GarageYrBlt_1900-1949', 'GarageYrBlt_1950-1969',
       'GarageYrBlt_1970-1989', 'GarageYrBlt_1990-2010', 'GarageYrBlt_No_data',
       'Heating_Floor', 'Heating_OthW', 'MasVnrArea_0-200',
       'MasVnrArea_1000-1200', 'MasVnrArea_1200-1400', 'MasVnrArea_1400-1600',
       'MasVnrArea_200-400', 'MasVnrArea_400-600', 'MasVnrArea_600-800',
       'MasVnrArea_800-1000', 'Utilities_NoSeWa']] = 0
```


```python
test_df_d.drop(['GarageYrBlt', 'MasVnrArea', 'Utilities_NA'], axis=1, inplace=True)
```


```python
test_df_d.shape
```




    (1459, 60)



Awesome we are ready to go


```python
# Normalize test dataframe
test_df_d[important_num_cols] = scaler.fit_transform(test_df_d[important_num_cols])
```


```python
# Set the model
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01)

# Fit the model on all train data (before split, U can try to train it on X_train, y_train)
xgb.fit(X, y) 
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
predictions = xgb.predict(test_df_d)
predictions

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
      <td>130554.250000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>152364.656250</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>185537.312500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>180152.515625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>187944.812500</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Export csv file
Submission.to_csv('submission.csv', index=False)
```
