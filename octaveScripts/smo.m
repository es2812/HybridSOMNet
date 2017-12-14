lines = dlmread('../originalFiles/digitos.test.normalizados.txt',' ')
longitud_fichero = length(lines(:,1))
tamanyo_entrada = length(lines(1,:))

#el fichero esta formateado como [entrada,empty line,salida,empty line] por lo que nos interesa
#cada cuarta linea de la matriz

indices_entradas = [1:4:longitud_fichero]
numero_instancias = length(indices_entradas)
entrada = lines(indices_entradas,:)

#se añade una dimension a cada instancia para evitar igualdad de vectores al normalizar
entrada = [entrada,ones(numero_instancias,1)]

#se calcula la norma de cada instancia como la raiz cuadrada de la suma de los cuadrados
#y se divide cada componente del vector por esta norma, tal quedan vectores de norma 1.
normas_entradas = sqrt(sum(entrada.^2,2))
entrada = entrada./normas_entradas 

#SMO. CONSTANTES:
n_iteraciones = 20
filas_smo = 12
columnas_smo = 8

n_neuronas = filas_smo*columnas_smo #tamaño del espacio de salida
tam_espacio_entrada = tamanyo_entrada+1#por la coordenada extra

#inicializamos pesos aleatorios (entre -5 y 5)
#p[i][j] = peso de la neurona i para la dimension del espacio de entrada j
pesos = rand(n_neuronas,tamanyo_entrada)
#normalizamos
pesos = [pesos,ones(n_neuronas,1)] #añadida coordenada extra
normas_pesos = sqrt(sum(pesos.^2,2))
pesos = pesos./normas_pesos

#d[i][j] = distancia de la instancia i a la neurona j
distancias = zeros(numero_instancias,n_neuronas)
i = 1
indice_ganadoras = zeros(numero_instancias,1)
#por cada entrada
while i<=numero_instancias #TODO: ufuncs
  j=1
  #por cada neurona
  while j<=n_neuronas
    distancias(i,j) = sum(entrada(i,:).*pesos(j,:))
    j++
  end
  [x,indice_ganadoras(i)] = min(distancias(i,:))
  i++
end

[x,indice_ganadoras2] = min(distancias,2)
#modulo del indice (ciclico)
#radio inicial para cubrir toda la dimension mas pequeña sin pisarse al ser ciclico
#alfa = alfa_0/(1+(t/n_muestras)) t AUMENTA POR CADA MUESTRA, PERO NO VUELVE A 0 TRAS FIN DE EPOCA
#alfa inicial no entre 0 y 1 a ojo (10, 20)
#arquitectura del mapa cuadrado.
#etiquetado por neuronas (necesitas las salidas)