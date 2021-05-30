---
title: "SLiM code for flucatuating N and s"
excerpt: "SLiM code for simulations with deterministic and stochastic fluctuations in population size and selection strength."
date: 2021-06-15
last_modified_at: false
header:
  overlay_image: /assets/images/05_2021/star_galaxy_1200x777.jpg
  overlay_filter: 0.5
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
---



```
initialize(){
	initializeMutationRate(1e-7); //uniform mutation rate per base per gen
	initializeMutationType("m1", 0.5, "f", 1e-4); //m1 = slightly  deleterious
	initializeGenomicElementType("g1", m1, 1); //g1 = coding site
	initializeGenomicElement(g1, 0, 99999); //Chromosome with 10000 bp
	initializeRecombinationRate(1e-8); //Assign recombination rate
}
1{	
	defineConstant("N", 1000); //Set current N as 1000
	sim.addSubpop("p1", N); //Create population with size = N
}
late(){
	//Fluctuate N every 100 generations
	if(sim.generation%100 == 0){		
		if(N == 1000){rm("N",T); defineConstant("N", 100);}
		else{rm("N",T); defineConstant("N", 1000);}
	}
	p1.setSubpopulationSize(asInteger(N)); //Set population size 
}
10000 late(){
	sim.outputFull(); //Output full data
}
```



```
1{	
	defineConstant("N", N1); //Set current N as N1
	sim.addSubpop("p1", N); //Create population with size = N
}
late(){
	//Fluctuate N every K generations
	if(sim.generation%K == 0){		
		if(N == N1){rm("N",T); defineConstant("N", N2);}
		else{rm("N",T); defineConstant("N", N1);}
	}
	p1.setSubpopulationSize(asInteger(N)); //Set population size 
}
```

```
slim -d K=100 -d N1=1000 -d N1=100 FlucN_Deterministic.txt
```



```
1{	
	defineConstant("N", N1); //Set current N as N1
	sim.addSubpop("p1", N); //Create population with subze = N
}
late(){
	//rho is prob env stays the same
	STAY = rbinom(1, 1, rho);
	//if STAY==0, then randomly choose env based on alpha
	if(STAY==0){
		ENV = rbinom(1, 1, 1-alpha); //Choose env
		if(ENV==0){rm("N",T); defineConstant("N", N1);}
		else{rm("N",T); defineConstant("N", N2);}
		p1.setSubpopulationSize(asInteger(N)); //Set population size 
	} 
}
```

```
slim -d rho=0.5 -d alpha=0.5 -d N1=1000 -d N1=100 FlucN_Stochastic.txt
```







```
initialize(){
	initializeMutationRate(1e-7); //uniform mutation rate per base per gen
	initializeMutationType("m1", 0.5, "f", S1); //m1 = slightly  deleterious
	initializeGenomicElementType("g1", m1, 1); //g1 = coding site
	initializeGenomicElement(g1, 0, 99999); //Chromosome with 10000 bp
	initializeRecombinationRate(1e-8); //Assign recombination rate
}
1{	
	defineConstant("S", S1); //set current S as S1
	sim.addSubpop("p1", N); //Create population with size = N
}
late(){
	//rho is prob env stays the same
	STAY = rbinom(1, 1, rho);
	//if STAY==0, then randomly choose env based on alpha
	if(STAY==0){
		ENV = rbinom(1, 1, 1-alpha); //Choose selection env
		if(ENV==0){rm("S",T); defineConstant("S", S1);}
		else{rm("S",T); defineConstant("S", S2);}
		
		mut = sim.mutationsOfType(m1); //get all m1 mutations
		mut.setSelectionCoeff(S); //Set selection strength for m1
	} 
}
10000 late(){
	sim.outputFull(); //Output full data
}
```



```
slim -d rho=0.5 -d alpha=0.5 -d N=1000 -d S1=0.001 -d S2=0.0001 SimpleFlucS_Stochastic.txt
```

