entrenamiento = csvread('training.csv');
#la ultima columna es la salida deseada.
num_muestras = size(entrenamiento)(1);
dimension = size(entrenamiento)(2)-1;
entrada = entrenamiento(1:num_muestras,1:dimension);
salida = entrenamiento(1:num_muestras,dimension+1);

#transformamos la salida por un vector con 0.9 en la posicion referente al digito y 0.1 en el resto
salida_deseada(1:num_muestras,1:10) = 0.1;
for i=1:num_muestras
  salida_deseada(i,salida(i)+1) = 0.9; #+1 por la indexacion por 1
endfor

capa_oculta = 20; #numero de neuronas en la capa oculta
capa_salida = 10; #numero de neuronas en la capa de salida
capa_entrada = dimension; #tendremos dimension neuronas en la capa de entrada
max_epocas = 3;
factor_aprendizaje=0.9;
bias=1;
factor_inercia=0.2;

%Normalización
xx = zeros(num_muestras,dimension);
for i=1:dimension
  xx(:,i) = (entrada(:,i)-min(entrada(:,i)))/(max(entrada(:,i))-min(entrada(:,i)));
endfor

#peso wij conecta la neurona j con la i
ww = (rand(capa_oculta,capa_entrada+1)*10)-5; #pesos capa oculta rand(-5,5) (+1 peso para el bias)
ws = (rand(capa_salida,capa_oculta+1)*10)-5; #pesos capa salida idem

%entrenamiento
yy = zeros(1,capa_oculta); #salida capa oculta

ys = zeros(1,capa_salida);#salida capa salida
ds = zeros(1,capa_salida); #termino delta capa salida

#guardamos un array con el incremento del peso de la muestra anterior, que aplicaremos
#a la iteracion actual para evitar caer en minimos locales
inercia_o = zeros(capa_oculta,capa_entrada+1);
inercia_s = zeros(capa_salida,capa_oculta+1);

for epoca=1:max_epocas
  for i=1:num_muestras
 
    %Fase hacia delante
    muestra_actual = [xx(i,:),bias]; #se añade un termino bias, que tendra su propio peso
    
    yy = 1./(1+exp(-(muestra_actual*ww'))); #salida de la capa oculta (sigmoide) 
    entrada_siguiente_capa = [yy,bias]; 
    ys = atan(entrada_siguiente_capa*ws'); #salida de la capa salida (arcotangente)
        
    %Fase hacia detras
    deseada_actual = salida_deseada(i,:);
    
    #capa de salida
    #cada neurona de la capa de salida tiene un termino delta
    #igual a la componente de la salida deseada correspondiente menos la salida de la neurona
    #multiplicado por la derivada de atan que es 1/(1+x^2)
    for j=1:capa_salida
      ds(j) = (deseada_actual(j) - ys(j))*(1/(1+ys(j)^2));
      #por cada conexion que entra en la neurona j actualizamos el peso
      for k=1:capa_oculta
        incremento_peso = (factor_aprendizaje*ds(j)*yy(k))+(factor_inercia*inercia_s(j,k));
        ws(j,k) += incremento_peso;
        inercia_s(j,k) = incremento_peso; #actualizamos la inercia para la siguiente muestra
      endfor
      #tenemos que modificar el peso del bias
      incremento_peso = (factor_aprendizaje*ds(j)*bias)+(factor_inercia*inercia_s(j,capa_oculta+1));
      ws(j,capa_oculta+1) += incremento_peso;
      inercia_s(j,capa_oculta+1) = incremento_peso;
    endfor
    #capa oculta
    #en la capa oculta el termino delta de cada neurona es el producto de matrices de 
    #la delta de la capa siguiente con los pesos que conectan la neurona con cada
    #neurona de la capa siguiente.
    #Ademas se multiplica por la derivada de la funcion sigmoide, F'(x) = F(x)(1-F(x))
    for j=1:capa_oculta
      dd = (ds*ws(:,j))*yy(j)*(1-yy(j)); #delta de la capa de entrada a la oculta, solo necesitamos valor puntual
      #por cada conexion que entra en la neurona j actualizamos el peso
      for k=1:capa_entrada
        incremento_peso = (factor_aprendizaje*dd*muestra_actual(k))+(factor_inercia*inercia_o(j,k));
        ww(j,k) += incremento_peso;
        inercia_o(j,k) = incremento_peso;
      endfor   
      #peso del bias
      incremento_peso = (factor_aprendizaje*dd*bias)+(factor_inercia*inercia_o(j,capa_entrada+1));
      ww(j,capa_entrada+1) += incremento_peso;
      inercia_o(j,capa_entrada+1) = incremento_peso;     
    endfor 
  endfor
endfor  
  
#TEST
test = csvread('test.csv');
#la ultima columna es la salida deseada.
num_muestras = size(test)(1);
dimension = size(test)(2)-1;
entrada = test(1:num_muestras,1:dimension);
salida = test(1:num_muestras,dimension+1);

%Normalización
xx = zeros(num_muestras,dimension);
for i=1:dimension
  xx(:,i) = (entrada(:,i)-min(entrada(:,i)))/(max(entrada(:,i))-min(entrada(:,i)));
endfor

fallos=0;
for i=1:num_muestras
  muestra_actual = [xx(i,:),bias];
  
   yy = 1./(1+exp(-(muestra_actual*ww'))); #salida de la capa oculta (sigmoide)  
   entrada_siguiente_capa = [yy,bias];
   ys = atan(entrada_siguiente_capa*ws'); #salida de la capa salida (arcotangente)
  
   [r,salida_obtenida] = max(ys); #rechazamos r, el valor, nos interesa el indice
   salida_obtenida -= 1; #por la indexacion por 1
   
   if(salida_obtenida != salida(i))
    fallos++;
   endif 
endfor
  
tasa_error = (fallos/num_muestras)*100;
printf("Tasa de error: %f %s\n",tasa_error,"%");



