package com.aliyun.codeupdemo.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import java.util.Random;

@RestController
public class P3cScannerController {

    @RequestMapping("/")
    public String index() {
        Random R = new Random();
        int randInt = R.nextInt();

        if (randInt % 2 == 0)
            return HelloWorld;

        return HELLO_WORLD;
    }

    private static final ExecutorService THREAD_EXECUTOR = Executors.newSingleThreadExecutor();

    private String HelloWorld = "Welcome to Codeup";

    private static final String HELLO_WORLD = "Nice to meet you";
}

