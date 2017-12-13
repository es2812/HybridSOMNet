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
n_iteraciones = 20000
filas_smo = 5
columnas_smo = 8

n_neuronas = filas_smo*columnas_smo #tamaño del espacio de salida
tam_espacio_entrada = tamanyo_entrada+1#por la coordenada extra

#inicializamos pesos aleatorios
pesos = rand(n_neuronas,tamanyo_entrada)
#normalizamos
pesos = [pesos,ones(n_neuronas,1)]


  