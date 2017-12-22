#ENTRENAMIENTO
lines = dlmread('../originalFiles/digitos.entrena.normalizados.txt',' ');
longitud_fichero = length(lines(:,1));
tamanyo_entrada = length(lines(1,:));
tamanyo_salida = 10;

#el fichero esta formateado como [entrada,empty line,salida,empty line] por lo que nos interesa
#cada cuarta linea de la matriz (EN LINUX)
indices_entradas = [1:4:longitud_fichero];
#en WINDOWS descomentar esta linea:
#indices_entradas = [1:2:longitud_fichero];
numero_instancias = length(indices_entradas);
entrada = (lines(indices_entradas,:) == 0.9); #se convierte la entrada a 0s y 1s 0.9->1

#se añade una dimension a cada instancia para evitar igualdad de vectores al normalizar
entrada = [entrada,ones(numero_instancias,1)];

#se calcula la norma de cada instancia y se divide cada componente del vector 
#por esta norma, tal que quedan vectores de norma 1.
i=1;
while i<=numero_instancias
  entrada(i,:) = entrada(i,:)./norm(entrada(i,:));
  i++;  
end


#SMO. CONSTANTES:
n_iteraciones = 20;
filas_smo = 12;
columnas_smo = 8;
radio_vecindad = 3;
#radio_vecindad = ceil(min(filas_smo,columnas_smo)/2)-1; #radio inicial cubre el maximo de la dimension mas pequeña sin overlap
alfa_inicial = 20; #alfa inicial (probar 20)

n_neuronas = filas_smo*columnas_smo; #tamaño del espacio de salida
tam_espacio_entrada = tamanyo_entrada+1; #por la coordenada extra

#inicializamos pesos aleatorios (entre -5 y 5)
#p[i][j] = peso de la neurona i para la dimension del espacio de entrada j
#no se añade en los pesos una columna extra de 1s
pesos = (rand(n_neuronas,tamanyo_entrada+1)*10)-5;

i=1;
while i<=n_neuronas
  pesos(i,:) = pesos(i,:)./norm(pesos(i,:));
  i++;
end

#todos los vectores entrada y pesos son unitarios, por lo que podemos utilizar
#el coseno del angulo entre ellos. El maximo coseno implicara la neurona mas cercana:
#
#   analiticamente: x w = producto escalar = sum(x_i*w_i)
#   geometricamente: x w = |x||w|cos(x,w) = 1*1*cos(x,w) = cos(x,w)
#
#por lo que podemos calcular el coseno del angulo como el producto escalar de x y w

epoca = 1;
t = 0;
while epoca <= n_iteraciones
  
  i = 1;
  #por cada muestra
  while i<=numero_instancias
  
    #se designa el alfa para esta muestra
    alfa = alfa_inicial/(1+(t/numero_instancias));
    
    muestra_actual = entrada(i,:);
     
    #necesitamos los cosenos solo durante la muestra actual, nos sirve un vector
    cosenos = zeros(n_neuronas,1);
    
    cosenos = muestra_actual*pesos';
    [x,ganadora] = max(cosenos); #x se desecha, es el valor, nos interesa el indice
    
    #modificamos el peso de la ganadora
    peso_no_normal = pesos(ganadora,:)+(muestra_actual.*alfa);
    pesos(ganadora,:) = (peso_no_normal)./norm(peso_no_normal);
    
    #BUCLE DE ENCUENTRO Y MODIFICACION DE VECINAS
    #solo necesario cuando el radio de vecindad es > 0, si R=0, solo necesitamos modificar la ganadora
    if radio_vecindad > 0
    
      #calculamos la fila y columna de la ganadora para hacer mas facil calculos siguientes
      #indice = (fila-1)*NUMCOL + columna
      #fila = ceil(indice/NUMCOL) (por el indexado por 1. Si fuera indexado por 0 seria floor)
      #columna = mod(indice,NUMCOL)+1 (idem)
      fila_ganadora = ceil(ganadora/columnas_smo);
      columna_ganadora = mod(ganadora,columnas_smo);
      #debido al indexado por 1, lo que matematicamente es la columna 0, en realidad es la ultima
      if columna_ganadora == 0
        columna_ganadora = columnas_smo;
      endif
      
      #se añade una fila a recorrer por cada unidad que aumenta el radio, mas la fila de la ganadora
      filas_a_recorrer = (radio_vecindad*2)+1;
      #idem
      columnas_a_recorrer = (radio_vecindad*2)+1;

      recorriendo_fila = fila_ganadora-radio_vecindad;
      fi=1;  
      while fi<=filas_a_recorrer
        recorriendo_columna = columna_ganadora-radio_vecindad ;
        co=1;
        while co<=columnas_a_recorrer
          #debido al caracter ciclico del mapa, se debe calcular el modulo
          fila_vecina = mod(recorriendo_fila,filas_smo);
          #debido al indexado por 1, lo que matematicamente es la columna/fila 0, en realidad es la ultima
          if fila_vecina == 0
            fila_vecina = filas_smo;
          endif
          columna_vecina = mod(recorriendo_columna,columnas_smo);
          if columna_vecina == 0
            columna_vecina = columnas_smo;
          endif
          #modificamos el peso de la neurona vecina
          indice_v = (fila_vecina-1)*columnas_smo + columna_vecina;
          #este metodo detecta tambien la propia neurona ganadora, que ya modificamos antes, la ignoramos
          if indice_v != ganadora
            peso_no_normal = pesos(indice_v,:)+(muestra_actual.*alfa);
            pesos(indice_v,:) = (peso_no_normal)./norm(peso_no_normal);
          endif
          recorriendo_columna++;
          co++;
        end
        recorriendo_fila++;
        fi++;
      end
    endif
    i++;
    t++;
  end
  #se modifica el radio de vecindad, acabara siendo 0
  if radio_vecindad > 0
    radio_vecindad --;
  end
  epoca++;
end

#Entrenamiento terminado.

#etiquetado por neuronas 
#Para linux
indices_salida = [3:4:longitud_fichero];
#en WINDOWS descomentar esta linea:
#indices_salida = [2:2:longitud_fichero];
#la salida esta formateada como un vector con tantos componentes con 0.9 en el 
#numero al que corresponde y el resto 0.1s
salida = lines(indices_salida,1:tamanyo_salida);
#obtenemos el digito que representa la salida
#salida_numerica = repmat([0:tamanyo_salida-1],numero_instancias,1)(salida==0.9);
salida_numerica = zeros(numero_instancias,1);
for i=1:numero_instancias
  [x,salida_numerica(i)] = max(salida(i,:));
  salida_numerica(i) -= 1; #los digitos van del 0-9
endfor

etiquetas = zeros(n_neuronas,1);
#se recorren todas las neuronas, se encuentra la muestra mas cercana, y se le aplica su clase
im = 1;
cosenos = zeros(numero_instancias,1);

while im<=n_neuronas
  peso_actual = pesos(im,:);
  
  cosenos = peso_actual*entrada';
  [x,muestra_ganadora] = max(cosenos);
  etiquetas(im) = salida_numerica(muestra_ganadora); #salida(i) corresponde a entrada(i)
  
  im++;
end
#etiquetado terminado 

etiquetas_print = reshape(etiquetas,columnas_smo,filas_smo)'

#MLP

#para cada muestra de entrada calculamos los cosenos resultantes y los pasamos de entrada a un mlp
entrada_mlp = zeros(numero_instancias,n_neuronas);
cosenos = zeros(n_neuronas,1);
i=1;
n=4;
while i<=numero_instancias

  muestra_actual=entrada(i,:);
  cosenos = muestra_actual*pesos';
  [x,ganadora] = max(cosenos);
  
  entrada_mlp(i,:) = cosenos';

  #filtramos la entrada para hacerla mas reconocible para mlp
  #escalado lineal, por cada entrada: e_i = e_i-max(e)/max(e)-min(e) 
  entrada_mlp(i,:) = (entrada_mlp(i,:)-min(entrada_mlp(i,:)))/(max(entrada_mlp(i,:))-min(entrada_mlp(i,:))); 
  
  i++;
end

#potenciado a la n
entrada_mlp=entrada_mlp.^n;

csvwrite('../mlp/training.csv',[entrada_mlp,salida_numerica])

#TEST
lines_test = dlmread('../originalFiles/digitos.test.normalizados.txt',' ');
longitud_fichero = length(lines_test(:,1));
#LINUX
indices_entradas = [1:4:longitud_fichero];
#en WINDOWS descomentar esta linea:
#indices_entradas = [1:2:longitud_fichero];
numero_instancias = length(indices_entradas);
entrada_test = (lines(indices_entradas,:)==0.9);

#normalizado de las entradas
entrada_test = [entrada_test,ones(numero_instancias,1)];
entrada_test = entrada_test./norm(entrada_test);

indices_salida = [3:4:longitud_fichero];
#en WINDOWS descomentar esta linea:
#indices_salida = [2:2:longitud_fichero];

salida_deseada = lines(indices_salida,:);

salida_numerica = zeros(numero_instancias,1);
for i=1:numero_instancias
  [x,salida_numerica(i)] = max(salida_deseada(i,:));
  salida_numerica(i) -= 1; #los digitos van del 0-9
endfor

salida_obtenida = zeros(numero_instancias,1);

entrada_mlp = zeros(numero_instancias,n_neuronas);

fallos = 0;
im = 1;
cosenos = zeros(n_neuronas,1);
#por cada muestra hallamos la neurona mas cercana, cuya etiqueta sera la clase de la entrada
while im<=numero_instancias

  muestra_actual=entrada_test(im,:);
    
  cosenos = muestra_actual*pesos';
  [x,ganadora] = max(cosenos);
  
  salida_obtenida(im) = etiquetas(ganadora);
  
  if(salida_obtenida(im) != salida_numerica(im))
    fallos++;
  endif
  
  #MLP
  entrada_mlp(im,:) = cosenos';
  
  #filtramos la entrada para hacerla mas reconocible para mlp
  #escalado lineal, por cada entrada: e_i = e_i-max(e)/max(e)-min(e) 
  entrada_mlp(im,:) = (entrada_mlp(im,:)-min(entrada_mlp(im,:)))/(max(entrada_mlp(im,:))-min(entrada_mlp(im,:)));
  im++;
end

entrada_mlp=entrada_mlp.^n;

tasa_error = (fallos/numero_instancias)*100;
printf("Tasa de error: %f %s\n",tasa_error,"%")

csvwrite('../mlp/test.csv',[entrada_mlp,salida_numerica]);