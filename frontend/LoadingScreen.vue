<template>
  <div id="loading">
    <img src="./assets/LoadingComponent.gif" />
  </div>
</template>

<script>
export default {
  name: "Loading"
};
</script>

<style>
#loading {
  position: absolute;
  background-color: white;
  height: 100vh;
  width: 100vw;
  text-align: center;
  line-height: 100vh;
}
</style>