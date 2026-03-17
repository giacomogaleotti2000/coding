package main

import "fmt"

// ritornarci sopra, studiare e ristudiare per fissare bene il concetto (molto potente!!!)

// interfaces are collection of method signatures
type employee interface {
	getName() string
	getSalary() int
}

func printEmployeeInfo(e employee) {
	fmt.Printf("name: %s\n", e.getName())
	fmt.Printf("salary: %d\n\n", e.getSalary())
}

// dipendente
type dipendente struct {
	nome    string
	cognome string
	eta     int
	RAL     int
}

func (dip dipendente) getName() string {
	return dip.nome
}
func (dip dipendente) getSalary() int {
	return dip.RAL
}

// ---

// partita iva
type partita_iva struct {
	nome               string
	piva               string
	costo_orario       int
	ore_annue_lavorate int
}

func (piv partita_iva) getName() string {
	return piv.nome
}
func (piv partita_iva) getSalary() int {
	return piv.costo_orario * piv.ore_annue_lavorate
}

func main() {

	dip1 := dipendente{
		nome:    "mario",
		cognome: "rossi",
		eta:     33,
		RAL:     38000,
	}

	piva1 := partita_iva{
		nome:               "ziopingu srl",
		piva:               "00440043020010",
		costo_orario:       35,
		ore_annue_lavorate: 700,
	}

	printEmployeeInfo(dip1)
	printEmployeeInfo(piva1)

}
