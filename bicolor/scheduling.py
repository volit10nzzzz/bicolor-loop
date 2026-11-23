"""两种题单交接方式，共用完全相同的出题和求解逻辑。"""  # 调度不参与奖励计算。
from concurrent.futures import ThreadPoolExecutor  # 管理常驻生产线程及其异常。
from queue import Queue, SimpleQueue  # 分别传递待解题单和已结算轮次。
from threading import Event  # 协调一个预取位置和提前退出。
from time import monotonic  # 只测量取到题单的等待时间。


def run_sync(rounds, prepare_batch, solve_batch):  # 同一线程轮流出题和解题。
    records = []  # 只保存已经结算的轮次。
    for index in range(rounds):  # 本轮求解结束后才进入下一轮。
        start = monotonic()  # 同步等待包含整段出题时间。
        batch = prepare_batch(index, tuple(records))  # Challenger 可以看见刚完成的上一轮。
        record = solve_batch(index, batch, records, monotonic() - start, None)  # 本批立即交给 Solver。
        records.append(record)  # 发布本轮完整反馈。
    return records  # 返回可回放的有序记录。


def run_async(rounds, prepare_batch, solve_batch):  # Challenger 生产题单，Solver 消费题单。
    ready = Queue(maxsize=1)  # 一个位置存放一整批已就绪的题。
    feedback = SimpleQueue()  # Solver 发布每轮已结算的记录。
    free_slot = Event()  # 一个预取位置包括正在准备和已经排队的题单。
    free_slot.set()  # 种子批可以立即开工。
    stopping = Event()  # 消费者提前退出时唤醒生产者。

    def produce():  # 常驻线程独立按计划轮次生产题单。
        snapshot = ()  # 局部快照只引用已经结算的记录。
        for planned in range(rounds):  # 最后一批交接后生产者退出。
            free_slot.wait()  # 上一题单被取走后才开始再准备一批。
            free_slot.clear()  # 单生产者独占这个位置。
            if stopping.is_set():  # 消费者发生错误时不再开工。
                return  # 已有调用由客户端正常记账。
            if planned == 1 or planned >= 3:  # Q1 等 H0，Q2 复用 H0，Q3 起各取下一条反馈。
                completed = feedback.get()  # 没有满足截止条件的反馈时休眠等待。
                if completed is None or stopping.is_set():  # 终止信号不会当作一轮反馈。
                    return  # 解除因反馈不足产生的等待。
                snapshot += (completed,)  # 本批只见到截止轮次及更早的原题。
            try:  # 模型错误必须能叫醒正在等题的 Solver。
                batch = prepare_batch(planned, snapshot)  # 使用冻结证据生成新题并混入历史题。
            except BaseException:  # 保留异常类型和原始堆栈。
                ready.put_nowait(None)  # 已预留的空位用来通知消费者。
                raise  # 异常由生产线程的 Future 保存。
            ready.put_nowait(batch)  # 单生产者已持有空位，整批原子交接。

    records = []  # 消费者拥有按求解顺序写入的总记录。
    with ThreadPoolExecutor(max_workers=1) as worker:  # 退出时回收已发起的推理。
        producer = worker.submit(produce)  # 全程只有一个 Challenger 生产任务。
        def check_producer():  # 每次新的 Solver 请求前检查后台是否失败。
            if producer.done():  # 正常情况下直接继续，不阻塞当前批次。
                producer.result()  # 若生产者失败则传播原始异常。
        try:  # 无论哪方失败都释放交接等待。
            for index in range(rounds):  # Solver 只按 FIFO 顺序领取。
                start = monotonic()  # 记录队列空时的实际等待。
                batch = ready.get()  # 已有题直接取，没题就等生产者。
                waited = monotonic() - start  # 后续求解时间不计入取题等待。
                if batch is None:  # 收到明确的生产异常通知。
                    producer.result()  # 此时等异常退出，不再发起求解。
                free_slot.set()  # 后台可在当前求解期间准备下一批。
                record = solve_batch(index, batch, records, waited, check_producer)  # 调用共享的 Solver 循环。
                records.append(record)  # 先完整结算再发布反馈。
                if index == 0 or index <= rounds - 3:  # 只发布尚有后续出题者需要的快照。
                    feedback.put(record)  # 生产者按轮读取，不访问正在写入的轨迹。
        finally:  # 正常完成和两端故障都能回收线程。
            stopping.set()  # 禁止开始新批次。
            feedback.put(None)  # 唤醒正在等观察的生产者。
            free_slot.set()  # 唤醒正在等队列位置的生产者。
    return records  # 返回已完成的全部轮次。
