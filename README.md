# radiology_app

### **3-5-2021**
L'obiettivo dell'app (per ora) è quello di analizzare immagini TAC di polmoni per esaminare la presenza o meno di COVID-19.


**CICLO DI VITA APP**

La homepage deve permettere di:
* caricare file di un *singolo paziente*
* caricare file di *molteplici pazienti*

Il flusso operativo sarà dunque distinto per le due opzioni sopra elencate, poichè: 
* nel primo si potrà seguire un iter più dettagliato nella parte di segmentazione/pre-processing dell'immagine, (*iter notebook*).
* nel secondo questo sarà il più automatico possibile, dunque cercando di evitare il più possibile l'intervento da parte del medico, (*iter notebook "nascosto"*). 

In entrambe però, una volta arrivati ad avere l* immagin* pront*, si farà previsione circa la presenza di COVID-19. 
Nel caso di un singolo paziente verrà quindi restitutita la confidenza di positività; mentre nel caso multi-paziente si restituirà: 
- tabella con confidenze di positività
- percentuale di positività

**N.B.** Questo è solamente il funzionamento di una prima versione dell'app. 

## ESECUZIONE

Per eseguire consiglio di installare prima un ambiente virtuale con:

``` virtualenv nome_env ```

Successivamente installare tutte le dipendenze con:

``` pip install -r requirements.txt ```

Successivamente per lanciare il server di sviluppo entrare nella directory **radiology** e lanciare:

``` python manage.py runserver ```

prova