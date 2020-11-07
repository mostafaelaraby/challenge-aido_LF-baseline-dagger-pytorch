
build:
	dts build_utils aido-container-build --ignore-untagged


push: build
	dts build_utils aido-container-push



submit:
	dts challenges submit


submit-bea:
	dts challenges submit --impersonate 1639 --challenge all --retire-same-label
