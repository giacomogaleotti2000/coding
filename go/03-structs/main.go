package main

import (
	"fmt"
)

// 1. first collection type of the course --> type that contains other types
type car struct {
	model  string
	heigth float32
	width  float32
}

// 2. structs can contain other structs like the example below
type factory struct {
	car_type        car
	production_line int
}

// 3. structs can contain other structs without adding the name layer like the example below
type factory_quiker struct {
	car             // to reference this, you just need factory_quiker.car_type instead of .car.car_type of the previous example
	production_line int
}

// 4. IMPORTANT --> go doesn't have the methods like python does (it has structs only), but the way it function with methods is shown here in this example.
type user struct {
	username string
	password string
}

func (u user) basicAuth() string { // this method can be called with u.basicAuth()
	return fmt.Sprintf("Authorization: Basic %s:%s", u.username, u.password)
}

func main() {
	fmt.Printf("first line\n")

	newcar := car{
		model:  "alfa romeo",
		heigth: 3.123,
	}

	fmt.Printf("%v\n", newcar)

	newfactory := factory{
		car_type:        newcar,
		production_line: 2,
	}

	fmt.Printf("%v\n", newfactory)
	newfactory.car_type.heigth = 3
	fmt.Printf("new factory heigth: %v\n", newfactory)
	fmt.Printf("car heigth: %f\n", newcar.heigth) // here you can see the "methods" in go, that you access through the dot.

	//anonymous structs --> structs that you need only in one instance
	anostruct := struct {
		ziopingu string
		number   int
	}{
		ziopingu: "ziopingui",
		number:   12345,
	}

	fmt.Printf("%v\n", anostruct)

	us := user{
		username: "giacomo",
		password: "analyst",
	}

	fmt.Printf("%s\n", us.basicAuth())
}
