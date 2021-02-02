def _name_to_int(name):
    integer=0
    if name=="Cook.Cleandishes":
        integer=1
    elif name=="Cook.Cleanup":
        integer=2
    elif name=="Cook.Cut":
        integer=3
    elif name=="Cook.Stir":
        integer=4
    elif name=="Cook.Usestove":
        integer=5
    elif name=="Cutbread":
        integer=6
    elif name=="Drink.Frombottle":
        integer=7
    elif name=="Drink.Fromcan":
        integer=8
    elif name=="Drink.Fromcup":
        integer=9
    elif name=="Drink.Fromglass":
        integer=10
    elif name=="Eat.Attable":
        integer=11
    elif name=="Eat.Snack":
        integer=12
    elif name=="Enter":
        integer=13
    elif name=="Getup":
        integer=14
    elif name=="Laydown":
        integer=15
    elif name=="Leave":
        integer=16
    elif name=="Makecoffee.Pourgrains":
        integer=17
    elif name=="Makecoffee.Pourwater":
        integer=18
    elif name=="Maketea.Boilwater":
        integer=19
    elif name=="Maketea.Insertteabag":
        integer=20
    elif name=="Pour.Frombottle":
        integer=21
    elif name=="Pour.Fromcan":
        integer=22
    elif name=="Pour.Fromcup":
        integer=23
    elif name=="Pour.Fromkettle":
        integer=24
    elif name=="Readbook":
        integer=25
    elif name=="Sitdown":
        integer=26
    elif name=="Takepills":
        integer=27
    elif name=="Uselaptop":
        integer=28
    elif name=="Usetablet":
        integer=29
    elif name=="Usetelephone":
        integer=30
    elif name=="Walk":
        integer=31
    elif name=="WatchTV":
        integer=32
    return integer


def _int_to_name(integer):
    if integer==0:
        label="Background"
    if integer==1:
        label="Cook.Clean_dishes"
    if integer==2:
        label="Cook.Cleanup"
    if integer==3:
        label="Cook.Cut"
    if integer==4:
        label="Cook.Stir"
    if integer==5:
        label="Cook.Usestove"
    if integer==6:
        label="Cutbread"
    if integer==7:
        label="Drink.Frombottle"
    if integer==8:
        label="Drink.Fromcan"
    if integer==9:
        label="Drink.Fromcup"
    if integer==10:
        label="Drink.Fromglass"
    if integer==11:
        label="Eat.Attable"
    if integer==12:
        label="Eat.Snack"
    if integer==13:
        label="Enter"
    if integer==14:
        label="Getup"
    if integer==15:
        label="Laydown"
    if integer==16:
        label="Leave"
    if integer==17:
        label="Makecoffee.Pourgrains"
    if integer==18:
        label="Makecoffee.Pourwater"
    if integer==19:
        label="Maketea.Boilwater"
    if integer==20:
        label="Maketea.Insertteabag"
    if integer==21:
        label="Pour.Frombottle"
    if integer==22:
        label="Pour.Fromcan"
    if integer==23:
        label="Pour.Fromcup"
    if integer==24:
        label="Pour.Fromkettle"
    if integer==25:
        label="Read book"
    if integer==26:
        label="Sitdown"
    if integer==27:
        label="Takepills"
    if integer==28:
        label="Uselaptop"
    if integer==29:
        label="Usetablet"
    if integer==30:
        label="Usetelephone"
    if integer==31:
        label="Walk"
    if integer==32:
        label="WatchTV"
    return label