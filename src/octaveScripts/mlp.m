function grafica_aciertos = mlp(entrenamiento,test,capa_oculta,max_epocas,factor_aprendizaje,factor_inercia)
  
  #la ultima columna es la salida deseada.
  num_muestras = size(entrenamiento)(1);
  dimension = size(entrenamiento)(2)-1;
  entrada = entrenamiento(1:num_muestras,1:dimension);
  salida = entrenamiento(1:num_muestras,dimension+1);
  
  num_muestras_t = size(test)(1);
  entrada_t = test(1:num_muestras_t,1:dimension);
  salida_t = test(1:num_muestras_t,dimension+1);
  
  xx = zeros(num_muestras,dimension);
  tt = zeros(num_muestras_t,dimension);
  %Estandarizar
  [xx,mu,sigma] = zscore(entrada,1);
  tt = (entrada_t-mu)./sigma;
  
  #transformamos la salida por un vector con 0.9 en la posicion referente al digito y 0.1 en el resto
  salida_deseada(1:num_muestras,1:10) = 0.1;
  for i=1:num_muestras
    salida_deseada(i,salida(i)+1) = 0.9; #+1 por la indexacion por 1
  endfor
  
  
  capa_salida = 10; #numero de neuronas en la capa de salida
  capa_entrada = dimension; #tendremos dimension neuronas en la capa de entrada
  tasa_aciertos_min = 100;
  bias=1;
  
  #peso wij conecta la neurona j con la i
  ww = (rand(capa_oculta,capa_entrada+1)*10)-5; #pesos capa oculta rand(-5,5) (+1 peso para el bias)
  ws = (rand(capa_salida,capa_oculta+1)*10)-5; #pesos capa salida idem
  
  yy = zeros(1,capa_oculta); #salida capa oculta
  
  ys = zeros(1,capa_salida);#salida capa salida
  ds = zeros(1,capa_salida+1); #termino delta capa salida +1 bias
  
  #guardamos un array con el incremento del peso de la muestra anterior, que aplicaremos
  #a la iteracion actual para evitar caer en minimos locales
  inercia_o = zeros(capa_oculta,capa_entrada+1);
  inercia_s = zeros(capa_salida,capa_oculta+1);
  
  grafica_error = zeros(max_epocas,1);
  
  for epoca=1:max_epocas
    %entrenamiento
    for i=1:num_muestras
   
      %Fase hacia delante
      muestra_actual = [xx(i,:),bias]; #se añade un termino bias, que tendra su propio peso
      
      yy = 1./(1+exp(-(muestra_actual*ww'))); #salida de la capa oculta (sigmoide) 
      #yy = tanh(muestra_actual*ww');
      input_v = [yy,bias]; 
      #ys = tanh(input_v*ws'); #salida de la capa salida (arcotangente)
      ys = 1./(1+exp(-(input_v*ws')));    
      
      %Fase hacia atras
      deseada_actual = salida_deseada(i,:);
      
      #capa de salida
      #cada neurona de la capa de salida tiene un termino delta
      #igual a la componente de la salida deseada correspondiente menos la salida de la neurona
      #multiplicado por la derivada de la sigmoide F, F'(x) = F(x)(1-F(x))
      ds = (deseada_actual - ys).*(ys.*(1-ys));
      for j=1:capa_salida      
        #por cada conexion que entra en la neurona j actualizamos el peso (incluido el del bias)
        incremento_peso = (factor_aprendizaje*ds(j)*input_v) + (factor_inercia*inercia_s(j,:));
        ws(j,:) += incremento_peso;
        inercia_s(j,:) = incremento_peso;
      endfor
      
      
      #capa oculta
      #en la capa oculta el termino delta de cada neurona es el producto de matrices de 
      #la delta de la capa siguiente con los pesos que conectan la neurona con cada
      #neurona de la capa siguiente.
      #Ademas se multiplica por la derivada de la funcion sigmoide F, F'(x) = F(x)(1-F(x))
      for j=1:capa_oculta
        #delta de la capa de entrada a la oculta, solo necesitamos valor puntual
        dd = (ds*ws(:,j))*yy(j)*(1-yy(j)); 
        #por cada conexion que entra en la neurona j actualizamos el peso
        incremento_peso = (factor_aprendizaje*dd*muestra_actual)+(factor_inercia*inercia_o(j,:));
        ww(j,:) += incremento_peso;
        inercia_o(j,:) = incremento_peso;
      endfor
      
    endfor
    
    
    %test
    aciertos=0;
    for i=1:num_muestras_t
      muestra_actual = [tt(i,:),bias];
      
      yy = 1./(1+exp(-(muestra_actual*ww'))); #salida de la capa oculta (sigmoide)  
      #yy = tanh(muestra_actual*ww');
      entrada_siguiente_capa = [yy,bias];
      #ys = tanh(entrada_siguiente_capa*ws'); #salida de la capa salida (arcotangente)
      ys = 1./(1+exp(-(entrada_siguiente_capa*ws')));
      
      [r,salida_obtenida] = max(ys); #rechazamos r, el valor, nos interesa el indice
      salida_obtenida -= 1; #por la indexacion por 1
       
      if(salida_obtenida == salida_t(i))
        aciertos++;
      endif     
    endfor
    
    tasa_aciertos = (aciertos/num_muestras_t)*100;
    grafica_aciertos(epoca) = tasa_aciertos;
  
    printf("epoca:%d aciertos: %f\n",epoca,tasa_aciertos);
    fflush(stdout);
    
    if(tasa_aciertos >= tasa_aciertos_min) #si se llega a la tasa de aciertos requerida las iteraciones terminan
      break;
    endif
    
  endfor  
endfunction