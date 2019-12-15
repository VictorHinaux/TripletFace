# Projet IA : 

Grâce à colab on importe le projet Git puis notre Drive:
```bash
!git clone https://github.com/VictorHinaux/TripletFace.git
from google.colab import drive
drive.mount('/content/drive')

```

Ensuite on importe les bibliothèques qui seront utilisées pour ce projet:
```bash
!pip3 install triplettorch
from triplettorch import HardNegativeTripletMiner
from triplettorch import AllTripletMiner

```
On extrait notre dataset de l'archive qui est sur notre Drive:

```bash
!unzip "/content/drive/My Drive/DatasetIA.zip"
```

Puis on entraine notre model. Ce model a fait 2 epochs car sinon il faut 5 à 6h pour l'entrainer sur les 10 epoch prévu de base.
```bash
!python3 -m tripletface.train -s dataset/ -m model -e 2
```

Après avoir fait tourner notre model, on obtient le fichier model.pt ainsi que deux visualisation:
### Première Epoch
![TripletFace/MODEL/vizualisation_0.png ](https://github.com/VictorHinaux/TripletFace/blob/master/MODEL/vizualisation_0.png)
### Deuxième Epoch
![TripletFace/MODEL/vizualisation_1.png ](https://github.com/VictorHinaux/TripletFace/blob/master/MODEL/vizualisation_1.png)

on utilise ensuite un jit compile:

```bash
from tripletface.core.model import Encoder
model = Encoder(64)
weights = torch.load( "/content/TripletFace/model/model.pt" )['model']
model.load_state_dict( weights )
jit_model = torch.jit.trace(model,torch.rand(8, 3, 3, 3)) 
torch.jit.save( jit_model, "/content/drive/My Drive/IA/jit_model.pt" )
```

## Difficulté et aller plus loin 

Pour faire ce projet je me suis heurté à un problème majeur: un manque de vocabulaire technique ainsi que de connaissance.
Pour m'en sortir j'ai essayé de lire plusieurs article, regarder plusieurs tutoriels mais sans aucun succès.
Je regrette de n'avoir pas réussi à mettre en place le code permettant de créer les centroïds et les Thesholds même si une partie de la solution semblait être facilement récupérable dans le code proposé par ce lien :

https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch

