# hzero (ES)

**hzero** es una librería de inferencia estadística enfocada en la implementación de contrastes de hipótesis, 
tanto paramétricos como no paramétricos. 
Está pensada como una herramienta educativa y experimental para explorar el comportamiento de diferentes tests 
estadísticos bajo diversas condiciones.

## 🚧 Estado del proyecto

Este proyecto está actualmente en desarrollo activo.

## ✨ Características principales

- Implementación de distribuciones usadas en contraste de hipótesis (como t de Student, chi-cuadrado, etc.)
- Soporte para contrastes paramétricos (media y varianza)
- Visualización de resultados con **matplotlib**
- Uso extensivo de **NumPy** para operaciones numéricas eficientes

> [!WARNING]
> La mayoría de las clases proporcionadas no han sido rigurosamente testeadas, lo que puede dar lugar a resultados inesperados en casos extremos.
> A lo largo del desarrollo de la librería se corregirán aquellos fallos que puedan surgir. También, puedes notificar la presencia
> de fallos como un issue.

## 📦 Tecnologías utilizadas

- [SciPy](https://scipy.org/)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

## 🔧 Instalación

> [!NOTE]
> Por el momento, no hay una forma estándar de instalar hzero. Para usarlo, puedes clonar este repositorio:

```bash
git clone https://github.com/eduvicgar/hzero.git
cd hzero
```

Luego puedes importar los módulos directamente desde tu entorno local de desarrollo.

## 📄 Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

## ✍️ Autor
eduvicgar

## 📌 Próximas mejoras
- Más contrastes de hipótesis paramétricos (diferencia de medias y cociente de varianzas)
- Constrastes de hipótesis no paramétricos como chi cuadrado de Pearson y Kolgomorov-Smirnoff
- Más tests automatizados
- Más documentación y notebooks de ejemplo
