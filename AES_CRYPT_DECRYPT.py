import numpy as np

S_box = np.array(
   [[" ",    "0",    "1",    "2",    "3",    "4",    "5",    "6",    "7",    "8",    "9",    "a",    "b",    "c",    "d",    "e",    "f"],
    ["0", "0x63", "0x7c", "0x77", "0x7b", "0xf2", "0x6b", "0x6f", "0xc5", "0x30", "0x01", "0x67", "0x2b", "0xfe", "0xd7", "0xab", "0x76" ],
    ["1", "0xca", "0x82", "0xc9", "0x7d", "0xfa", "0x59", "0x47", "0xf0", "0xad", "0xd4", "0xa2", "0xaf", "0x9c", "0xa4", "0x72", "0xc0" ],
    ["2", "0xb7", "0xfd", "0x93", "0x26", "0x36", "0x3f", "0xf7", "0xcc", "0x34", "0xa5", "0xe5", "0xf1", "0x71", "0xd8", "0x31", "0x15" ],
    ["3", "0x04", "0xc7", "0x23", "0xc3", "0x18", "0x96", "0x05", "0x9a", "0x07", "0x12", "0x80", "0xe2", "0xeb", "0x27", "0xb2", "0x75" ],
    ["4", "0x09", "0x83", "0x2c", "0x1a", "0x1b", "0x6e", "0x5a", "0xa0", "0x52", "0x3b", "0xd6", "0xb3", "0x29", "0xe3", "0x2f", "0x84" ],
    ["5", "0x53", "0xd1", "0x00", "0xed", "0x20", "0xfc", "0xb1", "0x5b", "0x6a", "0xcb", "0xbe", "0x39", "0x4a", "0x4c", "0x58", "0xcf" ],
    ["6", "0xd0", "0xef", "0xaa", "0xfb", "0x43", "0x4d", "0x33", "0x85", "0x45", "0xf9", "0x02", "0x7f", "0x50", "0x3c", "0x9f", "0xa8" ],
    ["7", "0x51", "0xa3", "0x40", "0x8f", "0x92", "0x9d", "0x38", "0xf5", "0xbc", "0xb6", "0xda", "0x21", "0x10", "0xff", "0xf3", "0xd2" ],
    ["8", "0xcd", "0x0c", "0x13", "0xec", "0x5f", "0x97", "0x44", "0x17", "0xc4", "0xa7", "0x7e", "0x3d", "0x64", "0x5d", "0x19", "0x73" ],
    ["9", "0x60", "0x81", "0x4f", "0xdc", "0x22", "0x2a", "0x90", "0x88", "0x46", "0xee", "0xb8", "0x14", "0xde", "0x5e", "0x0b", "0xdb" ],
    ["a", "0xe0", "0x32", "0x3a", "0x0a", "0x49", "0x06", "0x24", "0x5c", "0xc2", "0xd3", "0xac", "0x62", "0x91", "0x95", "0xe4", "0x79" ],
    ["b", "0xe7", "0xc8", "0x37", "0x6d", "0x8d", "0xd5", "0x4e", "0xa9", "0x6c", "0x56", "0xf4", "0xea", "0x65", "0x7a", "0xae", "0x08" ],
    ["c", "0xba", "0x78", "0x25", "0x2e", "0x1c", "0xa6", "0xb4", "0xc6", "0xe8", "0xdd", "0x74", "0x1f", "0x4b", "0xbd", "0x8b", "0x8a" ],
    ["d", "0x70", "0x3e", "0xb5", "0x66", "0x48", "0x03", "0xf6", "0x0e", "0x61", "0x35", "0x57", "0xb9", "0x86", "0xc1", "0x1d", "0x9e" ],
    ["e", "0xe1", "0xf8", "0x98", "0x11", "0x69", "0xd9", "0x8e", "0x94", "0x9b", "0x1e", "0x87", "0xe9", "0xce", "0x55", "0x28", "0xdf" ],
    ["f", "0x8c", "0xa1", "0x89", "0x0d", "0xbf", "0xe6", "0x42", "0x68", "0x41", "0x99", "0x2d", "0x0f", "0xb0", "0x54", "0xbb", "0x16" ]])

Sbox_inv = np.array([ 
    [" ",    "0",    "1",    "2",    "3",    "4",    "5",    "6",    "7",    "8",    "9",    "a",    "b",    "c",    "d",    "e",    "f"],
    ["0",  "0x52", "0x09", "0x6a", "0xd5", "0x30", "0x36", "0xa5", "0x38", "0xbf", "0x40", "0xa3", "0x9e", "0x81", "0xf3", "0xd7", "0xfb",],
    ["1",  "0x7c", "0xe3", "0x39", "0x82", "0x9b", "0x2f", "0xff", "0x87", "0x34", "0x8e", "0x43", "0x44", "0xc4", "0xde", "0xe9", "0xcb",],
    ["2",  "0x54", "0x7b", "0x94", "0x32", "0xa6", "0xc2", "0x23", "0x3d", "0xee", "0x4c", "0x95", "0x0b", "0x42", "0xfa", "0xc3", "0x4e",],
    ["3",  "0x08", "0x2e", "0xa1", "0x66", "0x28", "0xd9", "0x24", "0xb2", "0x76", "0x5b", "0xa2", "0x49", "0x6d", "0x8b", "0xd1", "0x25",],
    ["4",  "0x72", "0xf8", "0xf6", "0x64", "0x86", "0x68", "0x98", "0x16", "0xd4", "0xa4", "0x5c", "0xcc", "0x5d", "0x65", "0xb6", "0x92",],
    ["5",  "0x6c", "0x70", "0x48", "0x50", "0xfd", "0xed", "0xb9", "0xda", "0x5e", "0x15", "0x46", "0x57", "0xa7", "0x8d", "0x9d", "0x84",],
    ["6",  "0x90", "0xd8", "0xab", "0x00", "0x8c", "0xbc", "0xd3", "0x0a", "0xf7", "0xe4", "0x58", "0x05", "0xb8", "0xb3", "0x45", "0x06",],
    ["7",  "0xd0", "0x2c", "0x1e", "0x8f", "0xca", "0x3f", "0x0f", "0x02", "0xc1", "0xaf", "0xbd", "0x03", "0x01", "0x13", "0x8a", "0x6b",],
    ["8",  "0x3a", "0x91", "0x11", "0x41", "0x4f", "0x67", "0xdc", "0xea", "0x97", "0xf2", "0xcf", "0xce", "0xf0", "0xb4", "0xe6", "0x73",],
    ["9",  "0x96", "0xac", "0x74", "0x22", "0xe7", "0xad", "0x35", "0x85", "0xe2", "0xf9", "0x37", "0xe8", "0x1c", "0x75", "0xdf", "0x6e",],
    ["a",  "0x47", "0xf1", "0x1a", "0x71", "0x1d", "0x29", "0xc5", "0x89", "0x6f", "0xb7", "0x62", "0x0e", "0xaa", "0x18", "0xbe", "0x1b",],
    ["b",  "0xfc", "0x56", "0x3e", "0x4b", "0xc6", "0xd2", "0x79", "0x20", "0x9a", "0xdb", "0xC0", "0xfe", "0x78", "0xcd", "0x5a", "0xf4",],
    ["c",  "0x1f", "0xdd", "0xa8", "0x33", "0x88", "0x07", "0xc7", "0x31", "0xb1", "0x12", "0x10", "0x59", "0x27", "0x80", "0xec", "0x5f",],
    ["d",  "0x60", "0x51", "0x7f", "0xa9", "0x19", "0xb5", "0x4a", "0x0d", "0x2d", "0xe5", "0x7a", "0x9f", "0x93", "0xc9", "0x9c", "0xef",],
    ["e",  "0xa0", "0xe0", "0x3b", "0x4d", "0xae", "0x2a", "0xf5", "0xb0", "0xc8", "0xeb", "0xbb", "0x3c", "0x83", "0x53", "0x99", "0x61",],
    ["f",  "0x17", "0x2b", "0x04", "0x7e", "0xba", "0x77", "0xd6", "0x26", "0xe1", "0x69", "0x14", "0x63", "0x55", "0x21", "0x0c", "0x7d",]])

r_con = np.array([
    ["01","02","04","08","10","20","40","80","1b","36"],
    ["00","00","00","00","00","00","00","00","00","00"],
    ["00","00","00","00","00","00","00","00","00","00"],
    ["00","00","00","00","00","00","00","00","00","00"],
    ])
#la matrice du cryptage()
matrixMixColumnsEnceypt = np.array([ 
    [ "2", "3", "1", "1"],
    [ "1", "2", "3", "1"],
    [ "1", "1", "2", "3"],
    [ "3", "1", "1", "2"]
])
#la matrice du decryptage()
matrixMixColumnsDeceypt = np.array(
   [[0x0e ,  0x0b, 0x0d, 0x09],
    [0x09 ,  0x0e, 0x0b, 0x0d],
    [0x0d ,  0x09, 0x0e, 0x0b],
    [0x0b ,  0x0d, 0x09, 0x0e]]
    )

def subElement(str,sbox):
    x=0
    y=0
    if len(str) < 4:
        for i in sbox[:,0]:
            if(str[2]==i):
                break
            y += 1           
        x = 1  
    else:
        for i in sbox[:,0]:
            if(str[3]==i):
                break
            y += 1          
        for j in sbox[:,0]:
            if(str[2]==j):
                break
            x+=1  
    return(sbox[x][y])
#LES FONCTIONS DE CRYPTAGE
def sub_byte(A,sbox):
    temp=[]
    for i in range(0,4,1):
        for j in range(0,4,1):
            str= A[i][j]
            temp.append(subElement(str,sbox))
    B = np.array([temp,])
    B = np.reshape(B,(4,4))
    return B       

def add_round_key (state,key):
    l = []
    for i in range (0,4,1):#i boucle sur les ligne
        for j in range (0,4,1):#colone
            a = int(state[i][j],16)
            b = int(key[i][j],16)
            xor = hex(a^b)   
            l.append(xor)
    m_ex = np.array([l])
    m_ex = m_ex.reshape(4,4)
    return m_ex           

def chift_row_encrypt(state):
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    for i in range(0, 4, 1):
        for j in range (0, 4, 1):
            if i == 0:
                row1.append(state[i][j])
        if i == 1:
            row2.append(state[1][1])
            row2.append(state[1][2])
            row2.append(state[1][3])
            row2.append(state[1][0])
        if i == 2:
            row3.append(state[2][2])
            row3.append(state[2][3])
            row3.append(state[2][0])
            row3.append(state[2][1])
        if i == 3:
            row4.append(state[3][3])
            row4.append(state[3][0])
            row4.append(state[3][1])
            row4.append(state[3][2])
        newState = np.array([row1,row2,row3,row4])
    return newState

def mix_culumns_encrypt (state, matrix):
    reduceVal = "0x1b"
    resultState = []
    for k in range (0,4,1):
        A = state[0:4,k].tolist()
        for i in range(0, 4 ,1): 
            B = matrix[i, 0:4].tolist() 
            var = []  
            for j in range(0, 4 ,1):
                if B[j] == "1":
                    
                    temp = hex(int(A[j], 16))
                    var.append(temp)
                            
                elif B[j] == "2":
                    temp = hex(int("2", 16) * int(A[j], 16))
                    if len(temp) > 4:
                        temp = hex(int(temp, 16) ^ int(reduceVal, 16)) 
                    if len(temp) > 4:
                        temp = temp.replace("1","",1)
                    var.append(temp)
                           
                elif B[j] == "3":
                        temp1 =  hex(int("2", 16) * int(A[j], 16))
                        
                        temp = hex(int(temp1, 16) ^ int(A[j], 16))
                        
                        if len(temp) > 4:
                            temp = hex(int(temp, 16) ^ int(reduceVal, 16))                          
                        if len(temp) > 4:
                            temp = temp.replace("1","",1)
                        var.append(temp)                 
                if j == 3:    
                    x = hex (int(var[0],16))
                    for i in range(1,4,1):
                        x = hex(int(x,16) ^ int(var[i],16))                   
           
                    resultState.append(x)
            
    state = np.array([resultState])
    state = state.reshape(4,4)
    state = np.transpose(state)
    return state

#LES FONCTIONS DE DECRYPTAGE          

def chift_row_decrypt(state):
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    for i in range(0, 4, 1):
        for j in range (0, 4, 1):
            if i == 0:
                row1.append(state[i][j])
        if i == 1:
            row2.append(state[1][3])
            row2.append(state[1][0])
            row2.append(state[1][1])
            row2.append(state[1][2])
        if i == 2:
            row3.append(state[2][2])
            row3.append(state[2][3])
            row3.append(state[2][0])
            row3.append(state[2][1])
        if i == 3:
            row4.append(state[3][1])
            row4.append(state[3][2])
            row4.append(state[3][3])
            row4.append(state[3][0])
        newState = np.array([row1,row2,row3,row4])
    return newState

def gf2n_multiply( a,  b):
    overflow = 0x100
    modulus = 0x11B
    sum = 0
    a = int(a)
    b = int(b,16)
    while (b > 0):
        if (b & 1):
            sum = sum ^ a            # if last bit of b is 1, add a to the sum
        b = b >> 1    
                               # divide b by 2, discarding the last bit # divide b by 2, discarding the last bit
        a = a << 1                           # multiply a by 2
        if (a & overflow):
            a = a ^ modulus    # reduce a modulo the AES polynomial

    return sum

def mix_culumns_decrypt (state, matrix):
    
    resultState = []
    for k in range (0,4,1):
        A = state[0:4,k].tolist()
        for i in range(0, 4 ,1): 
            B = matrix[i, 0:4].tolist() 
            var = []  
            for j in range(0, 4 ,1):

                temp = gf2n_multiply(B[j],A[j])
                var.append(temp)
                               
                if j == 3:    
                    x = var[0]
                    for i in range(1,4,1):
                        x = x^ var[i]
  
                    resultState.append(hex(x))
            
    state = np.array([resultState])
    state = state.reshape(4,4)
    state = np.transpose(state)
    return state

#LES FONCTIONS DE GENERATION DE CLE
def sub_key_last_col(keyMatrix):
    sub_col = []
    temp = subElement(keyMatrix[1][3], S_box)
    sub_col.append(temp)
    temp = subElement(keyMatrix[2][3], S_box)
    sub_col.append(temp)
    
    temp = subElement((keyMatrix[3][3]), S_box)
    sub_col.append(temp)
    temp = subElement(keyMatrix[0][3], S_box)
    sub_col.append(temp)
    return sub_col
def rot_word(col,sub_col,keyIndex):
    #keyIndex pour indiquer la colonne utilisée dans rcon
    #selon la tour (chaque tour on utilise une clé)
    #CETTE FONTION FAIT SUB COL AND ROTWORD (FIRST COL)
    xr_con = r_con[0:4,keyIndex]
    new_col = []
    for i in  range(0,4,1):
        a = int(xr_con[i],16)
        b = int(col[i],16) 
        c = int(sub_col[i],16)
        new_col.append(hex(a^b^c)) 
            
    return new_col
def xorCol(col1,col2):
    newCol = []
    for i in  range(0,4,1):
        a= int(col1[i],16)
        b = int(col2[i],16)
        newCol.append(hex(a^b))      
    return newCol
def generation_key(key, keyIndex):
    newKeyList = []
    col1 = []
    for i in range (0,4,1):
        col1.append(key[i][0])
    
    temp = rot_word (col1,sub_key_last_col(key),keyIndex)
    
    newKeyList = temp
    for i in range (1,4,1):
        tempC = []
        for j in range(0, 4, 1):
            tempC.append(key[j][i])
        temp = xorCol(temp,tempC)
        newKeyList.extend(temp)
        
    newKey = np.array([newKeyList,])
    newKey = newKey.reshape(4,4)
    newKey = np.transpose(newKey)
    return newKey
def keys_generated (key):
    keys = [] #tableau des clés commence par 
              #la clé initial puis les clés générées
    key0 = key
    for i in range (0, 10, 1):
        key0 = generation_key(key0 , i)
        keys.append(key0)
    return keys

# ROUND ENCRYPTE
def initial_round(key,messageState):
    print("*************INITIAL ROUND*************")
    return add_round_key(key,messageState)
 
def main_rounds_encrypt(state, key):
    x = sub_byte(state,S_box)
    print("ByteSub\n",x)
    y = chift_row_encrypt(x)
    print("Chift row\n",y)
    z = mix_culumns_encrypt(y,matrixMixColumnsEnceypt)
    print("Mix columns \n",z)
    w = add_round_key(z,key)
    print("Add round key\n",w)
    return w

def final_round_encrypt(state, key):
    print("*************FINAL ROUND*************")
    x = sub_byte(state,S_box)
    print("Byte sub\n", x)
    y = chift_row_encrypt(x)
    print("Chift row\n", y)
    w = add_round_key(y,key)
    print("add round key\n", w)
    return w

#ROUNDS DECRYPTE

def main_rounds_decrypt(state, key):
    ## Encryption
    x = sub_byte(state,Sbox_inv)
    print("1\n",x)
    y = chift_row_decrypt(x)
    print("2\n",y)
    z = mix_culumns_decrypt(y,matrixMixColumnsDeceypt)
    print("3\n",z)
    w = add_round_key(z,key)
    print("4\n",w)  
    return w

def final_round_decrypt(state, key):
    print("*************FINAL ROUND*************")
    x = sub_byte(state,Sbox_inv)
    print("Byte sub\n", x)
    y = chift_row_decrypt(x)
    print("Chift row\n", y)
    w = add_round_key(y,key)
    print("add round key\n", w)
    return w

#CONVERSIONS

def ascii_to_hex(text):
    list = []
    for i in range(0,len(text),1):
        temp = hex(ord(text[i]))
        list.append(temp) 
    State = np.array([list])
    State = State.reshape(4,4)
    State = np.transpose(State)
    return State

def hex_to_ascii(hex_string):
    
    listR = []
    for i in range (0,4,1):
        for j in range (0,4,1):
            
            try:    
                hex_str = hex_string[i][j][2:]
                string = ''.join(chr(int(i, 16)) for i in hex_str.split())
                listR.append(string)
            except:
                print("Une exeption est produite") 

    result = ''.join(listR)
    
    return result


def ENCRYPTE(text, key):
    if len(key) != 16:
        print("Entrer une clé de 128 bits")
        return "Entrer une clé de 128 bits"
    else :
        
        if len(text) % 16 == 0:
            key = ascii_to_hex(key)
            state = ascii_to_hex(text)
            print("*************INITIAL STATE*************\n", state)
            print("*************INITIAL KEY*************\n",key)
            listKeys = keys_generated(key)
            state0 = initial_round(key, state)
            print(state0)
            listState = []
            listState.append(state0)
            for i in range( 0 ,9 , 1 ):
                print("*************Round",i+1)
                print("*************KEY\n",listKeys[i])
                state0 = main_rounds_encrypt(state0 , listKeys[i])
                listState.append(state0)

            chiperTextMatrix = final_round_encrypt(state0, listKeys[9])
            print(chiperTextMatrix)
            result = hex_to_ascii(chiperTextMatrix)
            
            return result
              
        else:
            print("Enter un message de taille égale à 128 bits")
            return "Enter un message de taille égale à 128 bits"


def DECRYPTE(text, key):
    if len(key) != 16:
        print("Entrer une clé de 128 bits")
        return "Entrer une clé de 128 bits"
    else :
        
        if len(text) % 16 == 0:
            key = ascii_to_hex(key)
            state = ascii_to_hex(text)
            print("*************INITIAL STATE*************\n", state)
            print("*************INITIAL KEY*************\n",key)
            listKeys = keys_generated(key)
            state0 = initial_round(listKeys[len(listKeys)-1], state)
            print(state0)
            listState = []
            listState.append(state0)
            for i in range( len(listKeys)-2 ,0 ,-1 ):
                print("*************Round",i+1)
                print("*************KEY\n",listKeys[i])
                state0 = main_rounds_decrypt(state0 , listKeys[i])
                listState.append(state0)

            chiperTextMatrix = final_round_decrypt(state0, key)
            result = hex_to_ascii(chiperTextMatrix)
            return result
              
        else:
            print("Enter un message de taille égale à 128 bits")
            return "Enter un message de taille égale à 128 bits"
            

from tkinter import * 

window = Tk()
window.geometry("720x480")
window.title("AES")
window.iconbitmap("index.ico")
window.config(bg="#4065A4")

def encrypteEntry():
    Inputstate = input1.get()
    Inputkey = input2.get() 
    result = ENCRYPTE(Inputstate , Inputkey)
    text.set(result)

def decrypteEntry():
    Inputstate = input1.get()
    Inputkey = input2.get() 
    result = DECRYPTE(Inputstate , Inputkey)
    text.set(result)

#creation du frame
frame = Frame(window,bg="#4065A4",)
frame.pack(expand=YES,)

input1= Entry(frame,font="courrier, 20", bg="#4065A4",fg="white" , justify=CENTER, width=40, textvariable=StringVar,)
input1.insert(0, 'Message')
input1.grid(padx=20, pady=10,)

input2= Entry(frame,font="courrier, 20", bg="#4065A4",fg="white" , justify=CENTER, width=40, textvariable=StringVar)
input2.insert(0, 'Clé')
input2.grid(padx=20, pady=10, )

button = Button(frame,font="courrier, 17", text= "crypter",command=encrypteEntry ,bg="white",fg="#4065A4" , justify=CENTER, width=30)
button.grid(padx=20, pady=10, )

button = Button(frame,font="courrier, 17", text= "Decrypter", command=decrypteEntry ,bg="white",fg="#4065A4" , justify=CENTER, width=30)
button.grid(padx=20, pady=10, )

text = StringVar()
text.set("Résultat")
label = Label(frame,font="courrier, 20",bg="#4065A4",fg="white" , justify=CENTER , width=40 , textvariable=text)

label.grid(padx=20, pady=10, )

window.mainloop()
