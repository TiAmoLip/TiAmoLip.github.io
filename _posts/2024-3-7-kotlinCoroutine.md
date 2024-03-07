---
layout:     post
title:      Kotlin Coroutine
# subtitle:   
date:       2024-03-07
author:     Placeholder
# header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - kotlin
    - coroutine
---

## 前言
这个是我几个月前时候学的kotlin coroutine入门级的东西，现在弄过来是为了复习一下。小小为教程引个流: https://www.bilibili.com/video/BV1VM41157ZK/?share_source=copy_web&vd_source=b9f49a5630f42d63aa1a8565bf07e848

顺序有点乱。

一个线程上可以运行多个协程，可以类比一下进程和线程的关系。

## Lesson 0
`suspend`关键字用来修饰一个函数，表示这个函数是一个挂起函数。如果不在协程作用域中调用就会报错。如果你在加了这个关键字的函数里加上一个Thread.sleep，那么他会提醒你。因为Thread.sleep是阻塞线程的，会导致整个线程上的协程停掉

一个最简单的协程实例是:
```kotlin
import kotlinx.coroutines.*

fun main () {
    CoroutineScope(Dispatchers.IO).launch{//可以认为CoroutineScope就是协程的作用域，然后括号里是协程执行在哪个线程里
        println("Hello from coroutines")
        delay(1000)
        println("End of coroutines")

    }
    println("Main")
    Thread.sleep(2000)
    println("Main End")
}
```

`Dispachers.IO`是一个调度器，表示协程运行在IO线程上，常用于针对网络请求和数据库之类的；`Dispatchers.Main`表示协程运行在主线程上，常用于更新UI；`Dispatchers.Default`表示协程运行在默认的线程上，常用于CPU密集型的操作。可以通过代码
```kotlin
Thread.currentThread().name
```
获得当前线程的名字。

## Lesson 1 runBlocking
```kotlin
runBlocking(context, block)
```
context管理协程运行在哪个线程上，block是一个lambda表达式，表示协程的执行体。在运行的时候，不论你的`CoroutineScope`采用哪个线程调度，他都会阻塞主线程。同时它的返回值是block的返回值。

而`launch`函数不会阻塞主线程，它的返回值是一个`Job`对象，可以用来取消协程，当然如果你是用`GlobalScope`来启动的话，它无法调用`cancel`。 launch不会等待协程返回

`WithContext`不会阻塞主线程，会阻塞父线程，但是会等待协程返回。
```kotlin
fun main () {
    CoroutineScope(Dispatchers.IO).launch {
        withContext(Dispatchers.Default) {
            for (i in 0..2) println("111111")
        }
        println("222222222222222222222222222222")

    }

    for (i in 1..2) {
        println("asdadasdadas")
    }
    Thread.sleep(5000)
}
```
输出结果:
```
asdadasdadas
asdadasdadas
111111
111111
111111
222222222222222222222222222222
```

所以如果你需要等待函数的返回值，就用`withContext`，如果不需要就用`launch`。

GlobalScope的声明周期是整个程序。以android程序为例，如果你在activity里面创建了一个GlobalScope，那么生命周期是整个应用程序。销毁activity都不行。而coroutinescope在activity创建的话，声明周期只在activity。

## Lesson 2 async
由于`WithContext`是串行的, 会导致效率低下。当两个任务没什么关系的时候，我们可以用async来并行执行，可以用`await`来等待返回值。
```kotlin
val c = async(start = CoroutineStart.LAZY) {
    delay(1000)
    10
}

val d = async(start = CoroutineStart.LAZY) {
        request()//它的返回值是100
    }
    println(d.await())// 拿到值

    print(
        select<String> {
            c.onAwait {
                "10"
            }
            d.onAwait {
                "100"
            }
        }
    )
```
除了`await`之外，还可以用`select`和`onAwait`实现一个回调函数。后者接受一个lambda函数，只要协程执行完就会执行这个回调。select简单来说就是里面接收多个协程，阻塞主线程，谁先弄完就把谁返回出去