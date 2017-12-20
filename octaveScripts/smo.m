#ENTRENAMIENTO
lines = dlmread('../originalFiles/digitos.entrena.normalizados.txt',' ');
longitud_fichero = length(lines(:,1));
tamanyo_entrada = length(lines(1,:));
tamanyo_salida = 10;

#el fichero esta formateado como [entrada,empty line,salida,empty line] por lo que nos interesa
#cada cuarta linea de la matriz

indices_entradas = [1:4:longitud_fichero];
numero_instancias = length(indices_entradas);
entrada = lines(indices_entradas,:);

#se añade una dimension a cada instancia para evitar igualdad de vectores al normalizar
entrada = [entrada,ones(numero_instancias,1)];

#se calcula la norma de cada instancia como la raiz cuadrada de la suma de los cuadrados
#y se divide cada componente del vector por esta norma, tal quedan vectores de norma 1.
normas_entradas = sqrt(sum(entrada.^2,2));
entrada = entrada./normas_entradas ;

#SMO. CONSTANTES:
n_iteraciones = 6;
filas_smo = 12;
columnas_smo = 8;
radio_vecindad = ceil(min(filas_smo,columnas_smo)/2)-1; #radio inicial ubre el maximo de la dimension mas pequeña sin overlap
alfa_inicial = 10; #alfa inicial (probar 20)

n_neuronas = filas_smo*columnas_smo; #tamaño del espacio de salida
tam_espacio_entrada = tamanyo_entrada+1; #por la coordenada extra

#inicializamos pesos aleatorios (entre -5 y 5)
#p[i][j] = peso de la neurona i para la dimension del espacio de entrada j
pesos = (rand(n_neuronas,tamanyo_entrada)*10)-5;

#normalizamos
pesos = [pesos,ones(n_neuronas,1)]; #añadida coordenada extra
normas_pesos = sqrt(sum(pesos.^2,2));
pesos = pesos./normas_pesos;

#todos los vectores entrada y pesos son unitarios, por lo que podemos utilizar
#el coseno del angulo entre ellos. El maximo coseno implicara la neurona mas cercana:
#
#   analiticamente: x w = producto escalar = sum(x_i*w_i)
#   geometricamente: x w = |x||w|cos(x,w) = 1*1*cos(x,w) = cos(x,w)
#
#por lo que podemos calcular el coseno del angulo como sum(x_i*w_i)

#necesitamos los cosenos solo durante la muestra actual, nos sirve un vector
cosenos = zeros(n_neuronas,1);

epoca = 1;
t = 0;
while epoca <= n_iteraciones
  i = 1;
  #por cada muestra
  while i<=numero_instancias
    
    #se designa el alfa para esta muestra
    alfa = alfa_inicial/(1+(t/numero_instancias));

    muestra_actual = entrada(i,:);
    
    cosenos = sum(muestra_actual.*pesos,2);    
    [x,ganadora] = max(cosenos); #x se desecha, es el valor, nos interesa el indice
    
    #modificamos el peso de la ganadora
    peso_no_normal = pesos(ganadora,:)+(muestra_actual.*alfa);
    normas_nuevo_peso = sqrt(sum(peso_no_normal.^2,2));
    pesos(ganadora,:) = (peso_no_normal)./(normas_nuevo_peso);
    
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
            normas_nuevo_peso = sqrt(sum(peso_no_normal.^2,2));
            pesos(indice_v,:) = (peso_no_normal)./(normas_nuevo_peso);
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
  if radio_vecindad>0
    radio_vecindad--;
  endif
  epoca++;
end

#Entrenamiento terminado.

#etiquetado por neuronas
indices_salida = [3:4:longitud_fichero];
#la salida esta formateada como un vector con tantos componentes con 0.9 en el 
#numero al que corresponde y el resto 0.1s
salida = lines(indices_salida,1:tamanyo_salida);
#obtenemos el digito que representa la salida
salida_numerica = repmat([0:tamanyo_salida-1],numero_instancias,1)(salida==0.9);

etiquetas = zeros(n_neuronas,1);
#se recorren todas las neuronas, se encuentra la muestra mas cercana, y se le aplica su clase
im = 1;
cosenos = zeros(numero_instancias,1);

while im<=n_neuronas
  peso_actual = pesos(im,:);
  cosenos = sum(entrada.*peso_actual,2); #vector con los cosenos entre todas las entradas y la neurona actual
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
while i<=numero_instancias

  muestra_actual=entrada(i,:);
  cosenos = sum(muestra_actual.*pesos,2);
  [x,ganadora] = max(cosenos);
  
  entrada_mlp(i,:) = cosenos';
  
  i++;
end

csvwrite('../mlp/training.csv',vertcat([1:n_neuronas+1],[entrada_mlp,salida_numerica]))

#TEST
lines_test = dlmread('../originalFiles/digitos.test.normalizados.txt',' ');
longitud_fichero = length(lines_test(:,1));

indices_entradas = [1:4:longitud_fichero];
numero_instancias = length(indices_entradas);
entrada_test = lines(indices_entradas,:);

#normalizado de las entradas
entrada_test = [entrada_test,ones(numero_instancias,1)];
normas_entradas = sqrt(sum(entrada_test.^2,2));
entrada_test = entrada_test./normas_entradas ;

indices_salida = [3:4:longitud_fichero];
salida_deseada = lines(indices_salida,:);
salida_deseada = repmat([0:tamanyo_salida-1],numero_instancias,1)(salida_deseada==0.9);
salida_obtenida = zeros(numero_instancias,1);

entrada_mlp = zeros(numero_instancias,n_neuronas);

fallos = 0;
im = 1;
cosenos = zeros(n_neuronas,1);
#por cada muestra hallamos la neurona mas cercana, cuya etiqueta sera la clase de la entrada
while im<=numero_instancias

  muestra_actual=entrada_test(im,:);
    
  cosenos = sum(muestra_actual.*pesos,2);
  [x,ganadora] = max(cosenos);
  
  entrada_mlp(im,:) = cosenos';
  
  
  salida_obtenida(im) = etiquetas(ganadora);
  
  if(salida_obtenida(im) != salida_deseada(im))
    fallos++;
  endif
  im++;
end

tasa_error = (fallos/numero_instancias)*100

#nombre_fichero = '../mapas/it';
#nombre_fichero = strcat(nombre_fichero,int2str(n_iteraciones));
#nombre_fichero2 = strcat(nombre_fichero,'.txt');
#nombre_fichero = strcat(nombre_fichero,'.csv');

#csvwrite(nombre_fichero,etiquetas_print);
#csvwrite(nombre_fichero2,tasa_error);

csvwrite('../mlp/test.csv',vertcat([1:n_neuronas+1],[entrada_mlp,salida_deseada]));