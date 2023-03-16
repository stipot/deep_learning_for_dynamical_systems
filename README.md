# Построение моделей на основе нейронных сетей для моделирования динамических систем

Это сопроводительный репозиторий к обзорной статье **Построение моделей на основе нейронных сетей для моделирования динамических систем**, в котором дается практическое описание того, как могут быть реализованы такие модели, как *нейронные обыкновенные дифференциальные уравнения* и *нейронные сети, основанные на физике*.
Полный документ доступен по адресу: https://dl.acm.org/doi/10.1145/3567591.

Код в репозитории реализован на Python с использованием PyTorch для определения и обучения моделей.
Сценарии можно запускать с параметрами по умолчанию для воспроизведения графиков, показанных в статье, а также таких вещей, как кривые потерь, которые были вырезаны из-за нехватки места:

<img src="docs/stream_plot.svg" width=49%><img src="docs/time_vs_state_plot.svg" width=49%>


<img src="docs/survey_structure.svg">


# Installing dependencies
Создать виртуальный env
```
python.exe -m venv env
./env/Scripts/activate.bat

```

Зависимости, необходимые для запуска скриптов, можно установить через pip с помощью файла `requirements.txt` следующим образом:
``` bash
python3 -m pip install -r requirements.txt
```
Если вы используете Conda, вы можете запустить следующую команду из новой среды:
``` bash
conda install --file requirements.txt
```

# Проведение экспериментов

Каждый эксперимент можно запустить с параметрами по умолчанию, выполнив скрипт в интерпретаторе Python следующим образом:
```
python3 experiments/<name of experiment>.py ...
```
В таблице ниже приведены команды, необходимые для обучения и оценки моделей, описанных в обзорной статье.

| Name                                         | Section | Command                                                         |
| -------------------------------------------- | ------- | --------------------------------------------------------------- |
| Vanilla Direct-Solution                      | 3.2     | python3 experiments/direct_solution.py --model vanilla          |
| Automatic Differentiation in Direct-Solution | 3.3     | python3 experiments/direct_solution.py --model autodiff         |
| Physics Informed Neural Networks             | 3.4     | python3 experiments/direct_solution.py --model pinn             |
| Hidden Physics Networks                      | 3.5     | python3 experiments/hidden_physics.py                           |
| Direct Time-Stepper                          | 4.2.1   | python3 experiments/time_stepper.py --solver direct             |
| Residual Time-Stepper                        | 4.2.2   | python3 experiments/time_stepper.py --solver resnet             |
| Euler Time-Stepper                           | 4.2.3   | python3 experiments/time_stepper.py --solver euler              |
| Neural ODEs Time-Stepper                     | 4.2.4   | python3 experiments/time_stepper.py --solver {rk4,dopri5,tsit5} |
| Neural State-Space Model                     | 4.3.1   | ...                                                             |
| Neural ODEs with input                       | 4.3.2-3 | ...                                                             |
| Lagrangian Time-Stepper                      | 4.4.1   | ...                                                             |
| Hamiltonian Time-Stepper                     | 4.4.1   | ...                                                             |
| Deep Potential Time-Stepper                  | 4.4.2   | ...                                                             |
| Deep Markov-Model                            | 4.5.1   | ...                                                             |
| Latent Neural ODEs                           | 4.5.2   | python3 experiments/latent_neural_odes.py                       |
| Bayesian Neural ODEs                         | 4.5.3   | ...                                                             |
| Neural SDEs                                  | 4.5.4   | ...                                                             |



#Докер-образ
Чтобы обеспечить возможность выполнения кода в будущем, мы предоставляем образ докера.
Образ Docker позволяет запускать код на виртуальной машине на базе Linux на любой платформе, поддерживаемой Docker.

Чтобы использовать образ докера, вызовите команду сборки в корне этого репозитория:
``` bash
docker build . -t python_dynamical_systems
```
После этого «containers», содержащие код и все зависимости, могут быть созданы с помощью команды «run»:
``` bash
docker run -ti python_dynamical_systems bash
docker run -it -v c:/3:/var/edata python_dynamical_systems bash
```
Команда установит интерактивное соединение с контейнером.
После этого вы можете выполнить код, как если бы он работал на вашем хост-компьютере:
``` bash
python3 experiments/time_stepper.py ...
```

# Ссылки

Если вы используете произведение, пожалуйста, рассмотрите возможность его цитирования:
``` bibtex
@article{10.1145/3567591,
author = {Legaard, Christian and Schranz, Thomas and Schweiger, Gerald and Drgo\v{n}a, J\'{a}n and Falay, Basak and Gomes, Cl\'{a}udio and Iosifidis, Alexandros and Abkar, Mahdi and Larsen, Peter},
title = {Constructing Neural Network Based Models for Simulating Dynamical Systems},
year = {2023},
issue_date = {November 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {55},
number = {11},
issn = {0360-0300},
url = {https://doi.org/10.1145/3567591},
doi = {10.1145/3567591},
abstract = {Динамические системы широко используются в естественных науках, таких как физика, биология и химия, а также в инженерных дисциплинах, таких как анализ цепей, вычислительная гидродинамика и управление. Для простых систем дифференциальные уравнения, управляющие динамикой, могут быть получены с применением фундаментальных физических законов. Однако для более сложных систем этот подход становится чрезвычайно трудным. Моделирование, управляемое данными, — это альтернативная парадигма, которая направлена на изучение аппроксимации динамики системы с использованием наблюдений за реальной системой. В последние годы возрос интерес к применению методов моделирования на основе данных для решения широкого круга задач в физике и технике. В этой статье представлен обзор различных способов построения моделей динамических систем с использованием нейронных сетей. В дополнение к основному обзору мы рассматриваем соответствующую литературу и выделяем наиболее важные проблемы численного моделирования, которые должна преодолеть эта парадигма моделирования. Основываясь на рассмотренной литературе и выявленных проблемах, мы проводим обсуждение перспективных направлений исследований.},
journal = {ACM Comput. Surv.},
month = {feb},
articleno = {236},
numpages = {34},
keywords = {physics-informed neural networks, physics-based regularization, Neural ODEs}
}
```
