addpath(genpath('octaveScripts/'));

n_iteraciones=20;
filas_som=12;
columnas_som=8;
alfa_inicial=20;

[tasa_error_som,training,test] = som(n_iteraciones,filas_som,columnas_som,alfa_inicial);
printf("Tasa de error del mapa: %f %s\n",tasa_error_som,"%");


capa_oculta = 20; 
max_epocas = 500;
factor_aprendizaje=0.3;
factor_inercia=0.3;

grafica_aciertos = mlp(training,test,capa_oculta,max_epocas,factor_aprendizaje,factor_inercia);

figure(1);
plot(grafica_aciertos,'-r','LineWidth',4);
grid on;
title('Evolucion de la tasa de aciertos en el tiempo');
xlabel('Epoca');
ylabel('% aciertos');