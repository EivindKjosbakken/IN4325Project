<template>
  <div class="container">
    <div class="search-container">
      <div class="search-input">
        <img src="./assets/GoogIR.png" width="150"/>
        <input class="search-bar" v-model="query" v-on:input="autosuggest(query)" v-on:keyup.enter="fetch(query, filters)" placeholder="Search for something" />
        <ul v-if="suggestions.length > 0">
          <li v-for="(item, index) in suggestions" :key="index">{{ item }}</li>
        </ul>
        <div class="search-button">
          <button class="button-primary" @click="fetch(query, filters)">Search</button>
        </div>
      </div>
    </div>
    <div class="filters" >
      <div class="dropdown">
        Filter by:
        <button class="dropbtn">Programme</button>
        <!-- <div class="dropdown-content">
          <label v-for="faculty in filterData.faculty"><input type="checkbox" v-model="filters.faculty" :value="faculty">{{ faculty }}</label>
        </div> -->
        <div class="dropdown-content">
          <label v-for="programme in filterData.programme"><input type="checkbox" v-model="filters.programme" :value="programme">{{ programme }}</label>
        </div>
      </div>
      <p v-if="timeTaken !== -1">Time elapsed: {{ timeTaken }} seconds</p>
    </div>
    <LoadingScreen v-if="isLoading" />
    <div class="search-result" v-else v-for="item in data">
      <a :href="item[1]" target="_blank">
        <h3 class="search-result-title" @click='window.location.href = item[1]'>{{ item[2] }}</h3>
      </a>
      <p class="search-result-abstract">{{ getLimitedAbstract(item[6]) }}</p>
    </div>
  </div>
</template>

<script>
import LoadingScreen from "./LoadingScreen";

export default {
  data() {
    return {
      query: '',
      suggestions: [],
      maxAbstractWords: 50,
      timeTaken: -1,
      isLoading: false,
      // Data index meanings
      // 0: uuid, 1: repository link, 2: title, 3: author, 4: contributor, 5: publication year, 6: abstract, 7: subject topic,
      // 8: language, 9: publication type, 10: publisher, 11: isbn, 12: issn, 13: patent, 14: patent status, 15: bibliographic note,
      // 16: access restriction, 17: embargo date, 18: faculty, 19: department, 20: research group, 21: programme, 22: project, 23: coordinates
      data: '',
      selectedUniversity: '',
      filters: {
        faculty: [],
        programme: []
      },
      filterData: {
        faculty: ['Aerospace Engineering', 'Applied Sciences', 'Architecture', 'Architecture and The Built Environment', 'Architecture and the Built Environment', 'Civil Engineering & Geosciences', 'Civil Engineering and Geosciences', 'Delft University of Technology', 'Electrical Engineering, Mathematics and Computer Science', 'Industrial Design Engineering', 'Mechanical, Maritime and Materials Engineering', 'OTB', 'OTB Research Institute', 'OTB Research Institute for the Built Environment', 'QuTech', 'Reactor Instituut Delft', 'Technology, Policy and Management'],
        programme: ['Aerospace Engineering', 'Applied Earth Sciences', 'Applied Geophysics', 'Applied Mathematics', 'Applied Physics', 'Applied Sciences', 'Architecture, Urbanism and Building Sciences', 'BioMedical Engineering', 'Biomedical Engineering', 'Chemical Engineering', 'Civil Engineering', 'Cognitive Robotics', 'Complex Systems Engineering and Management', 'Computer Engineering', 'Computer Science', 'Computer Science and Engineering', 'Computer Simulations for Science and Engineering', 'Cyber Security Group', 'Design for Interaction', 'Electrical Engineering', 'Engineering and Policy Analysis', 'European Master in Urbanism', 'European Wind Energy Masters', 'Geo-Energy Engineering', 'Geo-Engineering', 'Geomatics', 'Geoscience and Engineering', 'Geoscience and Remote Sensing', 'Geotechnical Engineering', 'Industrial Ecology', 'Integrated Product Design', 'Life Science and Technology', 'Management of Technology', 'Marine Technology', 'Materials Science and Engineering', 'Mechanical Engineering', 'Metropolitan Analysis, Design and Engineering', 'Offshore Engineering', 'Offshore and Dredging Engineering', 'Spacecraft Systems Engineering', 'Strategic Product Design', 'Sustainable Energy Technology', 'Technical Medicine', 'The Berlage Post-MSc in Architecture and Urban Design', 'Transport, Infrastructure and Logistics', 'Water Management', 'Water Resources Engineering']      }
    }
  },
  methods: {
    async fetch(query, filters) {
      this.isLoading = true;
      const options = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, filters })
      };
      
      fetch('http://localhost:5000/retrieve', options)
        .then(async response => {
          const data = await response.json();
          if (!response.ok) {
            const error = (data && data.message) || response.status;
            return Promise.reject(error);
          }
          this.timeTaken = data.time;
          this.data = data.results;
          this.isLoading = false;
        })
        .catch(error => {
          this.errorMessage = error;
          this.isLoading = false;
          console.error('There was an error!', error);
        });
    },
    autosuggest(query) {
      console.log(query)
      const options = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      };
      
      fetch('http://localhost:5000/autocomplete', options)
        .then(async response => {
          const data = await response.json();
          if (!response.ok) {
            const error = (data && data.message) || response.status;
            return Promise.reject(error);
          }
          // this.timeTaken = data.time;
          // this.data = data.results;
          // this.isLoading = false;
        })
        .catch(error => {
          this.errorMessage = error;
          this.isLoading = false;
          console.error('There was an error!', error);
        });
    }
  },
  computed: {
    getLimitedAbstract() {
      return function(abstract) {
        const words = abstract.split(" ");
        if (words.length > this.maxAbstractWords) {
          return words.slice(0, this.maxAbstractWords).join(" ") + " ...";
        } else {
          return abstract;
        }
      };
    }
  },
  components: {
    LoadingScreen
  }
}
</script>

<style>
.search-container {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
  background-color: #fff;
  border-radius: 5px;
}

.search-input {
  flex: 1;
  display: flex;
  align-items: center;
}

.search-bar {
  width: 80%;
  margin-left: 25px;
  border: none;
  font-size: 16px;
  padding: 10px;
  border-radius: 5px;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
}

.search-button {
  margin-left: 10px;
}

.search-button button {
  font-size: 16px;
  font-weight: 500;
  color: #fff;
  background-color: #3b82f6;
  border-radius: 5px;
  border: none;
  padding: 10px 20px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.search-button button:hover {
  background-color: #2563eb;
}

.filters {
  display: flex;
  font-size: 12px;
  max-width: 1000px;
  margin: 0 auto;
  padding-left: 20px;
  padding-right: 20px;
  background-color: #fff;
  border-radius: 5px;
}

.search-result {
  max-width: 1000px;
  margin: 0 auto;
  padding: 10px;
  background-color: #fff;
  border-radius: 5px;
}

.search-result-title {
  font-size: 26px;
  font-weight: bold;
}

.search-result-abstract {
  font-size: 1.2rem;
  color: #808080;
  margin-top: 0.5rem;
}

.dropdown {
  margin-left: 10px;
  margin-right: 20px;
  display: inline-block;
}

.dropdown-content {
  display: none;
  position: absolute;
  z-index: 1;
  background-color: #fff;
  min-width: 160px;
  box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
  border-radius: 5px;
  padding: 10px;
}

.dropdown-content label {
  display: block;
  margin-bottom: 10px;
}

.dropdown:hover .dropdown-content {
  display: block;
}

button {
  font-size: 16px;
  color: #fff;
  background-color: #3b82f6;
  border-radius: 5px;
  border: none;
  padding: 10px 20px;
  cursor: pointer;
  transition: background-color 0.3s ease;
  font-family: unset
}

button:hover {
  background-color: #2563eb;
}

.container {
  font-family: Verdana, Geneva, Tahoma, sans-serif;
}

.dropbtn {
  margin-left: 20px;
}
</style>
