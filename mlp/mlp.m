entrenamiento = csvread('training.csv');
#la primera fila se trata de etiquetas para los atributos.
#la ultima columna es la salida deseada.
num_muestras = size(entrenamiento)(1)-1;
dimension = size(entrenamiento)(2)-1;
entrada = entrenamiento(2:num_muestras+1,1:dimension);
salida = entrenamiento(2:num_muestras+1,dimension+1);

#transformamos la salida por un vector con 0.9 en la posicion referente al digito y 0.1 en el resto
salida_deseada(1:num_muestras,1:10) = 0.1;
for i=1:num_muestras
  salida_deseada(i,salida(i)+1) = 0.9; #+1 por la indexacion por 1
endfor

capa_oculta = 20; #numero de neuronas en la capa oculta
capa_salida = 10; #numero de neuronas en la capa de salida
capa_entrada = dimension; #tendremos dimension neuronas en la capa de entrada
max_epocas = 2000;
alfa=1;

%Normalización
xx = zeros(num_muestras,dimension);
for i=1:dimension
  xx(:,i) = (entrada(:,i)-min(entrada(:,i)))/(max(entrada(:,i))-min(entrada(:,i)));
endfor

ww = (rand(capa_entrada,capa_oculta)*10)-5; #pesos capa oculta rand(-5,5)
ws = (rand(capa_oculta,capa_salida)*10)-5; #pesos capa salida

deseada_actual = zeros(1,dimension);

%entrenamiento
#for i=1:num_muestras 
  
  yy = zeros(1,capa_oculta); #salida capa oculta
  dd = zeros(1,capa_oculta); #termino delta capa oculta

  ys = zeros(1,capa_salida);#salida capa salida
  ds = zeros(1,capa_salida); #termino delta capa salida
  
  %Fase hacia delante
  muestra_actual = entrada(i,:);
    
  yy = 1./(1+exp(-muestra_actual*ww)); #salida de la capa oculta (sigmoide)  
  ys = yy*ws; #salida de la capa salida (lineal)
        
  %Fase hacia detras
  deseada_actual = salida_deseada(i,:);
    
  #capa de salida
  #cada neurona de la capa de salida tiene un termino delta
  #igual a la componente de la salida deseada correspondiente menos la salida de la neurona 
  for j=1:capa_salida
    ds(j) = (deseada_actual(j) - ys(j));
    #por cada conexion que entra en la neurona j actualizamos el peso
    for k=1:capa_oculta
      ws(k,j) += alfa*ds(j)*yy(k);
    endfor
  endfor 
  #capa oculta
  #en la capa oculta el termino delta de cada neurona es el producto de matrices de 
  #la delta de la capa siguiente con los pesos que conectan la neurona con cada
  #neurona de la capa siguiente.
  #Ademas se multiplica por la derivada de la salida obtenida en esa neurona,
  #en la funcion de la capa de salida (lineal) esto es 1, pero en la capa oculta es
  #la derivada de la funcion sigmoide, F'(x) = F(x)(1-F(x))
  for j=1:capa_oculta
    dd(j) = (ds*ws(j,:)')*yy(j)*(1-yy(j));
    #por cada conexion que entra en la neurona j actualizamos el peso
    for k=1:capa_entrada
      ww(k,j) += alfa*dd(j)*muestra_actual(k);
    endfor            
  endfor 
  
#endfor




