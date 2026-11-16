import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p
        # TODO: реализовать формулу затухающего шага обучения
        #raise NotImplementedError()


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        X = self.model.X_train
        y = self.model.y_train
        tolerance = self.tolerance

        self.model.w = np.zeros(X.shape[1])
        self.iteration = 0
        self.model.loss_history = []
        loss_history = self.model.compute_loss()
        self.model.loss_history.append(loss_history)

        while (self.iteration < self.max_iter):
            delta = self._step()
            loss_history = self.model.compute_loss()
            self.model.loss_history.append(loss_history)

            if np.isnan(delta).any():
                break
            if delta @ delta < tolerance:
                break
    
        # TODO: implement
        # в конце также приcваивает атрибуту модели полученный loss_history


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать vanilla градиентный спуск
        # Можно использовать атрибуты класса self.model
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        w = self.model.w
        nu = self.lr_schedule.get_lr(self.iteration)
        w_new = w - nu * gradient
        diff = w_new - w
        self.model.w = w_new
        return diff
        #raise NotImplementedError()


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели
        X_train = self.model.X_train
        y_train = self.model.y_train
        w = self.model.w
        l = X_train.shape[0]
        i = np.random.randint(0, l, size=self.batch_size)
        x_batch = X_train[i]
        y_batch = y_train[i]
        gradient = self.model.compute_gradients(x_batch, y_batch)
        w_new = w - self.lr_schedule.get_lr(self.iteration) * gradient
        diff = w_new - w
        self.model.w = w_new
        return diff
        #raise NotImplementedError()


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать SAG
        X_train = self.model.X_train
        y_train = np.asarray(self.model.y_train).ravel()
        num_objects, num_features = X_train.shape
        w = self.model.w
        indecis = np.random.randint(0, num_objects, size=self.batch_size)

        if self.grad_memory is None or self.iteration == 0:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)
        
        for idx in indecis:
            g_new = self.model.compute_gradients(X_train[idx:idx+1], y_train[idx:idx+1])
            self.grad_sum += (g_new - self.grad_memory[idx]) / num_objects
            self.grad_memory[idx] = g_new
        w_new = w - self.lr_schedule.get_lr(self.iteration) * self.grad_sum
        diff = w_new - w
        self.model.w = w_new
        return diff
        #raise NotImplementedError()


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = np.asarray(self.model.y_train).ravel()
        w = self.model.w

        if self.velocity is None or self.iteration == 0:
            self.velocity = np.zeros_like(w)

        grad = self.model.compute_gradients(X_train, y_train)
        self.velocity = self.beta * self.velocity + self.lr_schedule.get_lr(self.iteration) * grad
        w_new = w - self.velocity
        diff = w_new - w
        self.model.w = w_new
        return diff

        # TODO: реализовать градиентный спуск с моментумом


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать Adam по формуле из ноутбука
        X_train = self.model.X_train
        y_train = np.asarray(self.model.y_train).ravel()
        w = self.model.w
        beta1 = self.beta1
        beta2 = self.beta2
        grad = self.model.compute_gradients(X_train, y_train)
        nu = self.lr_schedule.get_lr(self.iteration)

        if self.m is None or self.iteration == 0:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        k = self.iteration + 1
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)
        m_cup = self.m / (1 - beta1 ** k)
        v_cup = self.v / (1 - beta2 ** k)
        w_new = w - nu * m_cup / (np.sqrt(v_cup) + self.eps)
        diff = w_new - w
        self.model.w = w_new
        return diff
        raise NotImplementedError()


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        X = self.model.X_train
        y = self.model.y_train
        
        loss = self.model.loss_function
        self.model.w = loss.analytic_solution(X, y)

