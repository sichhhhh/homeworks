import numpy as np 
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC
from scipy.sparse.linalg import svds



class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

        

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        l = len(y)

        return 1/l * np.linalg.norm(X @ w - y)**2

       # raise NotImplementedError()
        # TODO: implement

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        #raise NotImplementedError()
        l = len(y)
        return 2/l * X.T @ (X@w - y)
        # TODO: implement

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса; 
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)

    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        #raise NotImplementedError()
        return np.linalg.inv(X.T @ X) @ X.T @ y
    
    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        u, s, vt = svds(X, k=min(X.shape)-1)
        w = vt.T @ np.linalg.pinv(np.diag(s)) @ u.T @ y
        return w

        #raise NotImplementedError()


class L2Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0,
                 analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate

        # analytic_solution_func is meant to be passed separately, 
        # as it is not linear to core solution
    
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        core_part = self.core_loss.loss(X, y, w)
        penalty_part = 0.5 * self.mu_rate * np.sum(w ** 2)
        return core_part + penalty_part

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.gradient(X, y, w)
        penalty_part = self.mu_rate * w
        return core_part + penalty_part

        raise NotImplementedError()
        # TODO: implement


class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \hat{y}
        """
        return X @ self.w
        # TODO: реализовать функцию предсказания в линейной регрессии
       # raise NotImplementedError("predict function is not implemented")

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        X = X_batch if X_batch is not None else self.X_train
        y = y_batch if y_batch is not None else self.y_train

        return self.loss_function.gradient(X, y, self.w)
        #raise NotImplementedError("Gradient caclucation is not implemented")


    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        X = X_batch if X_batch is not None else self.X_train
        y = y_batch if y_batch is not None else self.y_train
        return self.loss_function.loss(X, y, self.w)

        #raise NotImplementedError("Loss calculation is not implemented")


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.
        
        X: np.ndarray, 
        y: np.ndarray
        """
        # TODO: реализовать обучение модели
        self.X_train, self.y_train = X, y
        self.optimizer.optimize()

        #raise NotImplementedError("Linear Regression training is not implemented")


class LogCoshLoss(LossFunction):
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        w = np.asarray(w).ravel()
        z = X @ w - y
        return float(np.mean(np.logaddexp(z, -z) - np.log(2)))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        w = np.asarray(w).ravel()
        z = X @ w - y
        return X.T @ np.tanh(z) / len(y)


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        w = np.asarray(w).ravel()

        r = X @ w - y
        abs_r = np.abs(r)

        quadratic = 0.5 * np.minimum(abs_r, self.delta) ** 2
        linear = self.delta * np.maximum(abs_r - self.delta, 0.0)
        return float(np.mean(quadratic + linear))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        w = np.asarray(w).ravel()

        r = X @ w - y
        g = np.clip(r, -self.delta, self.delta)
        return X.T @ g / len(y)