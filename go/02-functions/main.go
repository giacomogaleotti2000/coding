package main

import "fmt"

// 1. you need to declare the type of each input and output
func concat(a string, b string) string {
	return a + " " + b
}

// 2. some more insights when there are multiple inputs of the same type and different outputs
func duplicate_name(nameA string) (string, string) {
	return nameA, nameA
}

// 3. consider that stuff is done automatically, for example in this function nothing is inside but two floats are returned
// this is still not reccomended as implicit stuff is harder to read and to understand later on.
func coords() (x float32, y float64) {
	// x and y are initialized at zero both

	return // the x and y are automatically returned if not specified
}

func main() {

	// first example of functions
	first_word := "hello"
	second_word := "world"
	fmt.Printf("I'm about to concatenate --> %s + %s <-- without the function\n", first_word, second_word)
	fmt.Printf("I'm about to concatenate --> %s <-- without the function\n", concat(first_word, second_word))

	// second example
	name := "pinguino"
	fmt.Printf("\nI'm just about to duplicate the name of %s\n", name)
	name1, name2 := duplicate_name(name)
	fmt.Printf("Duplicated name: %s %s\n", name1, name2)

	// third example
	x, y := coords()
	fmt.Printf("\nCoords: x: %f, y: %f\n", x, y)
}
